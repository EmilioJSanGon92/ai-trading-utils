import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def createHistPrices(start_date='2000-01-01', end_date='2024-05-01'):
    # Lista de tickers del S&P 500
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    sp500_tickers = [ticker for ticker in sp500_tickers if '.B' not in ticker]

    # Descargar precios históricos
    historical_prices = yf.download(sp500_tickers, start=start_date, end=end_date, auto_adjust=False)

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

def computingReturns(historical_prices, list_of_momentums):
    """
    Calcula retornos hacia adelante (forward returns) y momentums de distintos horizontes temporales.

    Parámetros:
    - historical_prices: DataFrame con precios ajustados. Columnas = tickers, filas = fechas.
    - list_of_momentums: lista de enteros con los días de momentum (por ej. [1, 5, 10])

    Retorna:
    - total_returns: DataFrame con retornos forward y momentums, índice = (ticker, fecha)
    """

    # Horizonte de predicción (para objetivo supervisado)
    forecast_horizon = 1

    # Calcular forward returns: % de cambio hacia adelante
    f_returns = historical_prices.pct_change(forecast_horizon, fill_method=None)
    f_returns = f_returns.shift(-forecast_horizon)  # Shift para alinear con la fecha de predicción
    f_returns = pd.DataFrame(f_returns.unstack())

    # Nombrar la columna según el horizonte
    name = f"F_{forecast_horizon}_d_returns"
    f_returns.rename(columns={0: name}, inplace=True)

    # Inicializar el DataFrame total con los forward returns
    total_returns = f_returns

    # Iterar sobre cada valor de momentum
    for i in list_of_momentums:
        # Calcular el momentum (retorno porcentual a i días)
        feature = historical_prices.pct_change(i, fill_method=None)
        feature = pd.DataFrame(feature.unstack())

        # Nombrar columna
        name = f"{i}_d_returns"
        feature.rename(columns={0: name}, inplace=True)

        # Asegurar que el índice tenga nombre de ticker (opcional si no está definido)
        feature.rename(columns={'level_0': 'Ticker'}, inplace=True)

        # Hacer merge con los retornos ya acumulados
        total_returns = pd.merge(
            total_returns,
            feature,
            left_index=True,
            right_index=True,
            how='outer'
        )

    # Eliminar filas con valores nulos (incompletas para entrenar)
    total_returns.dropna(axis=0, how='any', inplace=True)

    # Retornar el DataFrame limpio
    return total_returns
