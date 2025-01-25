import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, datetime
from scipy.stats import norm

# factor = rss / (n - m - 1)

def align(bS, nS, bD, nD):
    bitcoin = []
    nvidia = []
    crypto = bS['Close'].values.tolist()
    stock = nS['adjClose'].values.tolist()
    for i, j in enumerate(nD):
        if j in bD:
            k = bD.index(j)
            bitcoin.append(crypto[k])
            nvidia.append(stock[i])
    return np.array(bitcoin), np.array(nvidia)

def volatility(X, Y):
    volx, voly = [], []
    window = 200
    n = len(X)
    for i in range(window, n):
        volx.append(np.std(X[i-window:i]))
        voly.append(np.std(Y[i-window:i]))
    return volx, voly

def anova(X, y):
    Xh = np.array([[1.0, i, i**2] for i in X])
    Y = np.array(y)
    IXTX = np.linalg.inv(Xh.T.dot(Xh))
    n, m = Xh.shape
    m -= 1
    beta = IXTX.dot(Xh.T.dot(Y))
    Yhat = Xh @ beta
    rss = np.sum((Y - Yhat)**2)
    factor = rss / (n - m - 1)
    std_err = np.sqrt(np.diag(factor*IXTX))
    tstat = beta / std_err
    pvalues = [norm.cdf(t) if t < 0 else 1 - norm.cdf(t) for t in tstat]

    table = 'Variable: {} | Beta: {} | PValue: {}'
    output = 'ANOVA\n'
    for i, (b, p) in enumerate(zip(beta, pvalues)):
        output += table.format(i+1, b, p) + '\n'

    x0, x1 = np.min(X), np.max(X)
    dX = (x1 - x0)/99

    lx, ly = [], []
    for i in range(100):
        lx.append(x0 + i*dX)
        ly.append(beta[0] + beta[1]*lx[-1] + beta[2]*lx[-1]**2)
    
    return output, lx, ly

def anova3(X, y):
    Xh = np.array([[1.0, i, i**3] for i in X])
    Y = np.array(y)
    IXTX = np.linalg.inv(Xh.T.dot(Xh))
    n, m = Xh.shape
    m -= 1
    beta = IXTX.dot(Xh.T.dot(Y))
    Yhat = Xh @ beta
    rss = np.sum((Y - Yhat)**2)
    factor = rss / (n - m - 1)
    std_err = np.sqrt(np.diag(factor*IXTX))
    tstat = beta / std_err
    pvalues = [norm.cdf(t) if t < 0 else 1 - norm.cdf(t) for t in tstat]

    table = 'Variable: {} | Beta: {} | PValue: {}'
    output = 'ANOVA\n'
    for i, (b, p) in enumerate(zip(beta, pvalues)):
        output += table.format(i+1, b, p) + '\n'

    x0, x1 = np.min(X), np.max(X)
    dX = (x1 - x0)/99

    lx, ly = [], []
    for i in range(100):
        lx.append(x0 + i*dX)
        ly.append(beta[0] + beta[1]*lx[-1] + beta[2]*lx[-1]**3)
    
    return output, lx, ly

bitcoin = pd.read_csv('BTC-USD.csv')[::-1]
nvidia = pd.read_csv('NVDA.csv')[::-1]

btc_date = bitcoin['Time'].values.tolist()
nvd_date = nvidia['date'].values.tolist()

btc_date = list(map(lambda d: datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d'), btc_date))

x, y = align(bitcoin, nvidia, btc_date, nvd_date)

rorX = x[1:]/x[:-1] - 1.0
rorY = y[1:]/y[:-1] - 1.0

vx, vy = volatility(rorX, rorY)

out, lx, ly = anova(vx, vy)
out3, lx3, ly3 = anova3(vx, vy)

fig = plt.figure(figsize=(3, 6))
ax = fig.add_subplot(211)
ay = fig.add_subplot(212)

ax.scatter(vx, vy, color='red')
ax.plot(lx, ly, color='blue')
ax.set_title(out)
ax.set_xlabel('Volatility of Bitcoin')
ax.set_ylabel('Volatility of Nvidia')

ay.scatter(vx, vy, color='red')
ay.plot(lx3, ly3, color='blue')
ay.set_title(out3)
ay.set_xlabel('Volatility of Bitcoin')
ay.set_ylabel('Volatility of Nvidia')


plt.show()



