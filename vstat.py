import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, datetime
from scipy.stats import norm

# factor = rss / (n - m - 1)

# Align Nvidia and Bitcoin dataset
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

# Compute rolling volatility lists
def volatility(X, Y, cX, cY, cD):
    volx, voly = [], []
    cpbtc, cpnvd = [], []
    window = 200
    n = len(X)
    for i in range(window, n):
        volx.append(np.std(X[i-window:i]))
        voly.append(np.std(Y[i-window:i]))
    return volx, voly, cX[window+1:], cY[window+1:], cD[window+1:]

# Statistical Arbitrage function which trades on volatility relationships between Bitcoin and Nvidia
def StatArb(vx, vy, btc, nvda):
    alpha = 0.01
    window = 50
    n = len(vx)
    for i in range(window, n):
        # Build a rolling window for volatility for Bitcoin and Nvidia
        holdx = vx[i-window:i]
        holdy = vy[i-window:i]
        
        # Build a second order regression
        Xh = np.array([[1, k, k**2] for k in holdx])
        y = np.array(holdy)
        IXTX = np.linalg.inv(Xh.T.dot(Xh))

        # Compute the beta coeffecients and their respective PValues
        beta = IXTX.dot(Xh.T.dot(y))
        rss = sum((y - Xh.dot(beta))**2)
        factor = rss / (n - 2)
        stderr = np.sqrt(np.diag(factor*IXTX))
        tstat = beta / stderr
        pvalues = list(map(lambda u: norm.cdf(u) if u < 0.5 else 1 - norm.cdf(u), tstat))

        # If all three pvalues are significant, the ZScore of the spread, along with the price for both Bitcoin and Nvidia is yielded
        if pvalues[0] < alpha and pvalues[1] < alpha and pvalues[2] < alpha:
            spread = y - Xh.dot(beta)
            mu_spread = np.mean(spread)
            sd_spread = np.std(spread)/np.sqrt(window)
            Z = (spread[-1] - mu_spread)/sd_spread

            yield Z, btc[i], nvda[i]

# Load Bitcion and Nvidia datasets
bitcoin = pd.read_csv('BTC-USD.csv')[::-1]
nvidia = pd.read_csv('NVDA.csv')[::-1]

# Extract the dates for Bitcoin and Nvidia
btc_date = bitcoin['Time'].values.tolist()
nvd_date = nvidia['date'].values.tolist()

# Convert Bitcoin timestamp to date which matches Nvidias date format
btc_date = list(map(lambda d: datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d'), btc_date))

# Align the price data
x, y, dates = align(bitcoin, nvidia, btc_date, nvd_date)

# Compute the rate of returns of Bitcoin and Nvidia
rorX = x[1:]/x[:-1] - 1.0
rorY = y[1:]/y[:-1] - 1.0

# Fetch the rolling volatility list for Bitcoin and Nvidia
vx, vy, btc, nvda, xdates = volatility(rorX, rorY, x, y, dates)

# Initialize trading simulation parameters
position = 'neutral'
balanceBTC = 10000
balanceNVDA = 10000
txfee = 0.002
intrate = 0.0005

volbtc = 0
volnvda = 0
entrybtc = 0
entrynvda = 0

# Running the StatArb Backtest
for tstat, btc_price, nvda_price in StatArb(vx, vy, btc, nvda):

    # If you are long in Nvidia and the test statistic is greater than 3, exit long nvidia and exit short bitcoin
    if position == 'longnvda' and tstat > 3:
        position = 'neutral'
        balanceBTC += (entrybtc*(1-intrate)*(1-txfee) - btc_price*(1+txfee))*volbtc
        balanceNVDA += (nvda_price*(1-txfee) - entrynvda*(1+txfee))*volnvda
        print('Bitcoin Balance: {} | Nvidia Balance: {}'.format('{0:.2f}'.format(balanceBTC), '{0:.2f}'.format(balanceNVDA)))

    # If you are not in a position and the test statistic is less than -3, enter long nvidia short bitcoin
    if position == 'neutral' and tstat < -3:
        position = 'longnvda'
        volbtc = 0.9*balanceBTC / btc_price
        volnvda = int(0.9*balanceNVDA/nvda_price)
        entrybtc = btc_price
        entrynvda = nvda_price

    # If you are long Bitcoin and the test statistic is less than -3 exit long bitcoin and exit short nvidia
    if position == 'longbtc' and tstat < -3:
        position = 'neutral'
        balanceBTC += (btc_price*(1-txfee) - entrybtc*(1+txfee))*volbtc
        balanceNVDA += (entrynvda*(1-txfee)*(1-intrate) - nvda_price*(1-txfee))*volnvda
        print('Bitcoin Balance: {} | Nvidia Balance: {}'.format('{0:.2f}'.format(balanceBTC), '{0:.2f}'.format(balanceNVDA)))

    # If you are not in a position and the test statistic is greater than 3, enter long bitcoin and short nvidia
    if position == 'neutral' and tstat > 3:
        position = 'longbtc'
        volbtc = 0.9*balanceBTC / btc_price
        volnvda = int(0.9*balanceNVDA/nvda_price)
        entrybtc = btc_price
        entrynvda = nvda_price
    


    

