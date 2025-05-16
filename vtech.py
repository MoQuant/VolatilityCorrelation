import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, datetime
from scipy.stats import norm

# factor = rss / (n - m - 1)

# Aligns stock data with crypto data since stocks do not trade 24/7
def align(bS, nS, bD, nD):
    bitcoin = []
    nvidia = []
    crypto = bS['Close'].values.tolist()
    stock = nS['adjClose'].values.tolist()
    for i, j in enumerate(nD):
        if j in bD:
            # Index by date
            k = bD.index(j)
            bitcoin.append(crypto[k])
            nvidia.append(stock[i])
    return np.array(bitcoin), np.array(nvidia)

# Computes a rolling volatility window for 200 days
def volatility(X, Y):
    volx, voly = [], []
    window = 200
    n = len(X)
    for i in range(window, n):
        volx.append(np.std(X[i-window:i]))
        voly.append(np.std(Y[i-window:i]))
    return volx, voly

# ANOVA table designed to analyze non-linear regression between the two volatilities (2nd order)
def anova(X, y):
    # Declare matrix for non-linear regression
    Xh = np.array([[1.0, i, i**2] for i in X])
    Y = np.array(y)

    # Inverse of XTX
    IXTX = np.linalg.inv(Xh.T.dot(Xh))

    # Regression Dimensions
    n, m = Xh.shape
    m -= 1 # Subtract intercept degrees of freedom

    # Calculate the beta coeffecients
    beta = IXTX.dot(Xh.T.dot(Y))
    
    # Calculate YHat
    Yhat = Xh @ beta

    # Compute the standard error of each beta value and their respective pvalues
    rss = np.sum((Y - Yhat)**2)
    factor = rss / (n - m - 1)
    std_err = np.sqrt(np.diag(factor*IXTX))
    tstat = beta / std_err
    pvalues = [norm.cdf(t) if t < 0 else 1 - norm.cdf(t) for t in tstat]

    # Build ANOVA table
    table = 'Variable: {} | Beta: {} | PValue: {}'
    output = 'ANOVA\n'
    for i, (b, p) in enumerate(zip(beta, pvalues)):
        output += table.format(i+1, b, p) + '\n'

    # Build Non-Linear Regression Line
    x0, x1 = np.min(X), np.max(X)
    dX = (x1 - x0)/99

    lx, ly = [], []
    for i in range(100):
        lx.append(x0 + i*dX)
        ly.append(beta[0] + beta[1]*lx[-1] + beta[2]*lx[-1]**2)

    # Return ANOVA table and regression
    return output, lx, ly

# ANOVA table set to non-linear regression to 3rd order
def anova3(X, y):
    # 3rd order regression
    Xh = np.array([[1.0, i, i**3] for i in X])
    Y = np.array(y)
    IXTX = np.linalg.inv(Xh.T.dot(Xh))

    # Dimensions with degrees of freedom reduction
    n, m = Xh.shape
    m -= 1

    # Beta and Standard Error Computation with respective PValues
    beta = IXTX.dot(Xh.T.dot(Y))
    Yhat = Xh @ beta
    rss = np.sum((Y - Yhat)**2)
    factor = rss / (n - m - 1)
    std_err = np.sqrt(np.diag(factor*IXTX))
    tstat = beta / std_err
    pvalues = [norm.cdf(t) if t < 0 else 1 - norm.cdf(t) for t in tstat]

    # Building ANOVA table
    table = 'Variable: {} | Beta: {} | PValue: {}'
    output = 'ANOVA\n'
    for i, (b, p) in enumerate(zip(beta, pvalues)):
        output += table.format(i+1, b, p) + '\n'

    # Construct Regression Line
    x0, x1 = np.min(X), np.max(X)
    dX = (x1 - x0)/99

    lx, ly = [], []
    for i in range(100):
        lx.append(x0 + i*dX)
        ly.append(beta[0] + beta[1]*lx[-1] + beta[2]*lx[-1]**3)

    # Return table and regression line
    return output, lx, ly

# Load Bitcon and Nvidia data
bitcoin = pd.read_csv('BTC-USD.csv')[::-1]
nvidia = pd.read_csv('NVDA.csv')[::-1]

# Extract stock time and crypto time
btc_date = bitcoin['Time'].values.tolist()
nvd_date = nvidia['date'].values.tolist()

# Convert crypto dates to match stock date format
btc_date = list(map(lambda d: datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d'), btc_date))

# Align the stock and crypto dataset
x, y = align(bitcoin, nvidia, btc_date, nvd_date)

# Compute the rate of returns for Nvidia and Bitcoin
rorX = x[1:]/x[:-1] - 1.0
rorY = y[1:]/y[:-1] - 1.0

# Generate rolling volatility lists for Nvidia and Bitcoin
vx, vy = volatility(rorX, rorY)

# Generate the ANOVA tables for the 2nd and 3rd order regression
out, lx, ly = anova(vx, vy)
out3, lx3, ly3 = anova3(vx, vy)

# Initialize plots
fig = plt.figure(figsize=(3, 6))
ax = fig.add_subplot(211)
ay = fig.add_subplot(212)

# Plot the 2nd order regression
ax.scatter(vx, vy, color='red')
ax.plot(lx, ly, color='blue')
ax.set_title(out)
ax.set_xlabel('Volatility of Bitcoin')
ax.set_ylabel('Volatility of Nvidia')

# Plot the 3rd order regression
ay.scatter(vx, vy, color='red')
ay.plot(lx3, ly3, color='blue')
ay.set_title(out3)
ay.set_xlabel('Volatility of Bitcoin')
ay.set_ylabel('Volatility of Nvidia')


plt.show()



