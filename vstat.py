import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, datetime
from scipy.stats import norm

# factor = rss / (n - m - 1)

def align(bS, nS, bD, nD):
    bitcoin = []
    nvidia = []
    crypto = bS['Close'].values.tolist()
    stock = nS['adjClose'].values.tolist()
    sdates = nS['date'].values.tolist()
    dates = []
    for i, j in enumerate(nD):
        if j in bD:
            k = bD.index(j)
            bitcoin.append(crypto[k])
            nvidia.append(stock[i])
            dates.append(sdates[i])
    return np.array(bitcoin), np.array(nvidia), dates


def volatility(X, Y, cX, cY, cD):
    volx, voly = [], []
    cpbtc, cpnvd = [], []
    window = 200
    n = len(X)
    for i in range(window, n):
        volx.append(np.std(X[i-window:i]))
        voly.append(np.std(Y[i-window:i]))
    return volx, voly, cX[window+1:], cY[window+1:], cD[window+1:]

def StatArb(vx, vy, btc, nvda):
    alpha = 0.01
    window = 50
    n = len(vx)
    for i in range(window, n):
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
        pvalues = list(map(lambda u: norm.cdf(u) if u < 0.5 else 1 - norm.cdf(u), tstat))

        if pvalues[0] < alpha and pvalues[1] < alpha and pvalues[2] < alpha:
            spread = y - Xh.dot(beta)
            mu_spread = np.mean(spread)
            sd_spread = np.std(spread)/np.sqrt(window)
            Z = (spread[-1] - mu_spread)/sd_spread

            yield Z, btc[i], nvda[i]

bitcoin = pd.read_csv('BTC-USD.csv')[::-1]
nvidia = pd.read_csv('NVDA.csv')[::-1]

btc_date = bitcoin['Time'].values.tolist()
nvd_date = nvidia['date'].values.tolist()

btc_date = list(map(lambda d: datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d'), btc_date))

x, y, dates = align(bitcoin, nvidia, btc_date, nvd_date)

rorX = x[1:]/x[:-1] - 1.0
rorY = y[1:]/y[:-1] - 1.0

vx, vy, btc, nvda, xdates = volatility(rorX, rorY, x, y, dates)

position = 'neutral'
balanceBTC = 10000
balanceNVDA = 10000
txfee = 0.002
intrate = 0.0005

volbtc = 0
volnvda = 0
entrybtc = 0
entrynvda = 0

for tstat, btc_price, nvda_price in StatArb(vx, vy, btc, nvda):

    if position == 'longnvda' and tstat > 30:
        position = 'neutral'
        balanceBTC += (entrybtc*(1-intrate)*(1-txfee) - btc_price*(1+txfee))*volbtc
        balanceNVDA += (nvda_price*(1-txfee) - entrynvda*(1+txfee))*volnvda
        print('Bitcoin Balance: {} | Nvidia Balance: {}'.format('{0:.2f}'.format(balanceBTC), '{0:.2f}'.format(balanceNVDA)))

    if position == 'neutral' and tstat < -30:
        position = 'longnvda'
        volbtc = 0.9*balanceBTC / btc_price
        volnvda = int(0.9*balanceNVDA/nvda_price)
        entrybtc = btc_price
        entrynvda = nvda_price

    
    if position == 'longbtc' and tstat < -30:
        position = 'neutral'
        balanceBTC += (btc_price*(1-txfee) - entrybtc*(1+txfee))*volbtc
        balanceNVDA += (entrynvda*(1-txfee)*(1-intrate) - nvda_price*(1-txfee))*volnvda
        print('Bitcoin Balance: {} | Nvidia Balance: {}'.format('{0:.2f}'.format(balanceBTC), '{0:.2f}'.format(balanceNVDA)))

    if position == 'neutral' and tstat > 30:
        position = 'longbtc'
        volbtc = 0.9*balanceBTC / btc_price
        volnvda = int(0.9*balanceNVDA/nvda_price)
        entrybtc = btc_price
        entrynvda = nvda_price
    


    

