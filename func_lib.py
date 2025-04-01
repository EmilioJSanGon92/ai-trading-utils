import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def createHistPrices(start_date='2000-01-01', end_date='2024-05-01'):
    # Lista de tickers del S&P 500
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    sp500_tickers = [ticker for ticker in sp500_tickers if '.B' not in ticker]

    # Descargar precios históricos
    historical_prices = yf.download(sp500_tickers, start=start_date, end=end_date)

    # Filtrar solo columnas "Adj Close"
    historical_prices = historical_prices.loc[:, historical_prices.columns.get_level_values(0) == 'Adj Close']

    # Quitar multi-índice
    historical_prices.columns = historical_prices.columns.droplevel(0)

    # Requisitos mínimos por ticker
    MIN_REQUIRED_NUM_OBS_PER_TICKER = 100
    ticker_counts = historical_prices.count()
    valid_tickers_mask = ticker_counts >= MIN_REQUIRED_NUM_OBS_PER_TICKER
    valid_tickers = ticker_counts[valid_tickers_mask].index

    # Filtrar tickers válidos
    filtered_prices = historical_prices[valid_tickers]

    return filtered_prices
