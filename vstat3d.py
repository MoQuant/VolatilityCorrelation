import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, datetime
from scipy.stats import norm

# factor = rss / (n - m - 1)

# Align the Bitcoin and Nvidia datasets based on date
def align(bS, nS, bD, nD):
    bitcoin = []
    nvidia = []
    crypto = bS['Close'].values.tolist()
    stock = nS['adjClose'].values.tolist()
    sdates = nS['date'].values.tolist()
    dates = []
    for i, j in enumerate(nD):
        if j in bD:
            # Index by date
            k = bD.index(j)
            bitcoin.append(crypto[k])
            nvidia.append(stock[i])
            dates.append(sdates[i])
    return np.array(bitcoin), np.array(nvidia), dates

# Generate rolling volatility lists
def volatility(X, Y, cX, cY, cD):
    volx, voly = [], []
    cpbtc, cpnvd = [], []
    window = 200
    n = len(X)
    for i in range(window, n):
        # Rolling window of returns for Bitcoin and Nvidia
        volx.append(np.std(X[i-window:i]))
        voly.append(np.std(Y[i-window:i]))
    return volx, voly, cX[window+1:], cY[window+1:], cD[window+1:]

# StatArb Backtest Simulation
def StatArb(vx, vy, btc, nvda):
    alpha = 0.01
    window = 50
    n = len(vx)
    for i in range(window, n):
        # Build regression parameters with the betas and their respective pvalues
        holdx = vx[i-window:i]
        holdy = vy[i-window:i]
        Xh = np.array([[1, k, k**2] for k in holdx])
        y = np.array(holdy)
        IXTX = np.linalg.inv(Xh.T.dot(Xh))
        beta = IXTX.dot(Xh.T.dot(y))
        rss = sum((y - Xh.dot(beta))**2)
        factor = rss / (n - 2)
        stderr = np.sqrt(np.diag(factor*IXTX))
        tstat = beta / stderr
        pvalues = list(map(lambda u: norm.cdf(u) if u < 0 else 1 - norm.cdf(u), tstat))

        # If all pvalues are significant, yield the spreads zscore signal along with Bitcoin and Nvidia prices
        if pvalues[0] < alpha and pvalues[1] < alpha and pvalues[2] < alpha:
            spread = y - Xh.dot(beta)
            mu_spread = np.mean(spread)
            sd_spread = np.std(spread)/np.sqrt(window)
            Z = (spread[-1] - mu_spread)/sd_spread

            yield Z, btc[i], nvda[i]

# Load the datasets
bitcoin = pd.read_csv('BTC-USD.csv')[::-1]
nvidia = pd.read_csv('NVDA.csv')[::-1]

# Pull dates
btc_date = bitcoin['Time'].values.tolist()
nvd_date = nvidia['date'].values.tolist()

# Convert Bitcoin's date to match Nvidia's date
btc_date = list(map(lambda d: datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d'), btc_date))

# Align datasets
x, y, dates = align(bitcoin, nvidia, btc_date, nvd_date)

# Rate of Return of Bitcoin and Nvidia
rorX = x[1:]/x[:-1] - 1.0
rorY = y[1:]/y[:-1] - 1.0

# Rolling Volatility Lists
vx, vy, btc, nvda, xdates = volatility(rorX, rorY, x, y, dates)

# Entry and exit test statisitcs
entry_tstat = np.arange(2.0, 15.0, 1)#[1.96, 2.33, 3.5, 5.0, 10.0]
exit_tstat = np.arange(2.0, 15.0, 1)#[1.96, 2.33, 3.5, 5.0, 10.0]

# Build entry and exit test statistics into grid
tex, tey = np.meshgrid(entry_tstat, exit_tstat)

# Generate a plot figure and 3D subplot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

profit = []

for ii in range(len(entry_tstat)):
    tempz = []
    for jj in range(len(exit_tstat)):
        # Get Entry and Exit Test Statistics
        GEnter = tex[ii, jj]
        GExit = tey[ii, jj]

        # Reset trading parameters
        position = 'neutral'
        balanceBTC = 10000
        balanceNVDA = 10000
        txfee = 0.002
        intrate = 0.0005

        volbtc = 0
        volnvda = 0
        entrybtc = 0
        entrynvda = 0

        # Print the row and column indecies
        print(ii, jj)
        for tstat, btc_price, nvda_price in StatArb(vx, vy, btc, nvda):

            # Close long Nvidia short Bitcoin if GExit is reached
            if position == 'longnvda' and tstat > GExit:
                position = 'neutral'
                balanceBTC += (entrybtc*(1-intrate)*(1-txfee) - btc_price*(1+txfee))*volbtc
                balanceNVDA += (nvda_price*(1-txfee) - entrynvda*(1+txfee))*volnvda
                #print('Bitcoin Balance: {} | Nvidia Balance: {}'.format('{0:.2f}'.format(balanceBTC), '{0:.2f}'.format(balanceNVDA)))

            # Enter long Nvidia and short Bitcoin if -GEnter is reached
            if position == 'neutral' and tstat < -GEnter:
                position = 'longnvda'
                volbtc = 0.9*balanceBTC / btc_price
                volnvda = int(0.9*balanceNVDA/nvda_price)
                entrybtc = btc_price
                entrynvda = nvda_price

            # Exit long Bitcoin short Nvidia if -GExit is reached
            if position == 'longbtc' and tstat < -GExit:
                position = 'neutral'
                balanceBTC += (btc_price*(1-txfee) - entrybtc*(1+txfee))*volbtc
                balanceNVDA += (entrynvda*(1-txfee)*(1-intrate) - nvda_price*(1-txfee))*volnvda
                #print('Bitcoin Balance: {} | Nvidia Balance: {}'.format('{0:.2f}'.format(balanceBTC), '{0:.2f}'.format(balanceNVDA)))

            # Enter long Bitcoin short Nvidia if GEnter is reached
            if position == 'neutral' and tstat > GEnter:
                position = 'longbtc'
                volbtc = 0.9*balanceBTC / btc_price
                volnvda = int(0.9*balanceNVDA/nvda_price)
                entrybtc = btc_price
                entrynvda = nvda_price

        # Do not add negative balances to profit output
        tempz.append(max(balanceBTC, 0) + max(balanceNVDA, 0))

    # Add each simulation to profit list
    profit.append(tempz)

# Plot the profit of the stat arb runs in the 3D plot
profit = np.array(profit)

ax.plot_surface(tex, tey, profit, cmap='jet')
ax.set_title('Stat Arb on Volatility Results')
ax.set_xlabel('Entry TStat')
ax.set_ylabel('Exit TStat')
ax.set_zlabel('Portfolio Balance')

plt.show()

            

