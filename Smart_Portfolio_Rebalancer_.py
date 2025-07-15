import yfinance as yf
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import scipy.optimize as sco

portfolio = {
    'AAPL': 0.4,
    'MSFT': 0.3,
    'GOOGL': 0.2,
    'AMZN': 0.1
}

rebalance_threshold = 0.05
lookback_period = '2y'
rebalance_frequency = 30

def fetch_price_data(tickers, period):
    data = yf.download(tickers, period=period)['Adj Close']
    return data.dropna()

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = returns / std
    return returns, std, sharpe

def max_sharpe_ratio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def neg_sharpe(weights):
        return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = sco.minimize(neg_sharpe, num_assets * [1. / num_assets], bounds=bounds, constraints=constraints)
    return result.x

def fetch_sentiment(ticker):
    analyzer = SentimentIntensityAnalyzer()
    headlines = [
        f"{ticker} announces strong earnings",
        f"{ticker} faces regulatory scrutiny",
        f"Analysts upgrade {ticker}"
    ]
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    return np.mean(scores)

def adjust_weights_with_sentiment(weights):
    sentiment_scores = {ticker: fetch_sentiment(ticker) for ticker in weights.index}
    sentiments = pd.Series(sentiment_scores)
    scaled = MinMaxScaler(feature_range=(0.8, 1.2)).fit_transform(sentiments.values.reshape(-1, 1))
    scaled_sentiments = pd.Series(scaled.flatten(), index=sentiments.index)
    new_weights = weights * scaled_sentiments
    return new_weights / new_weights.sum()

def simulate_strategy(prices, rebalance_days):
    returns = prices.pct_change().dropna()
    portfolio_values = []
    weights = pd.Series(portfolio)
    current_value = 1.0

    for i in range(0, len(returns), rebalance_days):
        segment = returns.iloc[i:i + rebalance_days]
        mean_returns = segment.mean()
        cov_matrix = segment.cov()
        opt_weights = max_sharpe_ratio(mean_returns, cov_matrix)
        weights = pd.Series(opt_weights, index=prices.columns)
        weights = adjust_weights_with_sentiment(weights)
        for _, row in segment.iterrows():
            current_value *= (1 + np.dot(weights, row))
            portfolio_values.append(current_value)

    return pd.Series(portfolio_values, index=returns.index[rebalance_days-1::rebalance_days][:len(portfolio_values)])

def plot_performance(series):
    plt.figure(figsize=(12, 6))
    plt.plot(series, label="Strategy Value")
    plt.title("Portfolio Value Over Time with Sentiment-Based Rebalancing")
    plt.ylabel("Portfolio Value")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.show()

prices = fetch_price_data(list(portfolio.keys()), lookback_period)
strategy_result = simulate_strategy(prices, rebalance_frequency)
plot_performance(strategy_result)
