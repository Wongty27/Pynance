import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(ticker: str, start_date: str=None, end_date: str=None) -> tuple[pd.Series, pd.DataFrame]:
    data = yf.download(tickers=ticker, start=start_date, end=end_date)
    close_data = data['Close']
    returns = close_data.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

def get_weights(meanReturns) -> np.ndarray:
    weights = np.random.random(len(meanReturns))
    weights /= np.sum(weights)
    return weights

def simulation(
    n_sim: int,
    timeframe: int,
    weights: np.ndarray,
    meanReturns: pd.Series,
    covMatrix: pd.Dataframe,
    nitialPortfolio: float=10000
) -> np.ndarray:
    meanM = np.full(shape=(timeframe, len(weights)), fill_value=meanReturns)
    meanM = meanM.T

    portfolio_sims = np.full(shape=(timeframe, len(n_sim)), fill_value=0.0)

    for i in range(0, n_sim):
        Z = np.random.normal(shape=(timeframe, len(weights)))
        L= np.linalg.cholesky(covMatrix)
        dailyReturns = meanM + np.inner(L, Z)
        portfolio_sims[:, i] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

def plot_result(portfolio_sims: np.ndarray) -> None:
    plt.plot(portfolio_sims)
    plt.ylabel('Portfolio value ($)')
    plt.xlabel('Days')
    plt.title('MC Simulation of a Stock Portfolio')
    plt.show()