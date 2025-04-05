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


def compute_BM_Perf(total_returns):
    """
    Computes benchmark performance using equal-weighted daily mean returns.
    Calculates cumulative returns, calendar year returns, CAGR, and Sharpe ratio.
    
    Parameters:
    -----------
    total_returns : DataFrame
        Pandas DataFrame containing daily returns for multiple assets, 
        with a multi-index (Date, Ticker) and column 'F_1_d_returns'.
    
    Returns:
    --------
    cum_returns : DataFrame
        Cumulative returns over time.
    
    calendar_returns : DataFrame
        Calendar year (annual) returns.
    """

    # --- 1. Compute equal-weighted daily mean of all stocks ---
    daily_mean = pd.DataFrame(
        total_returns.loc[:, 'F_1_d_returns']
        .groupby(level='Date')
        .mean()
    )
    daily_mean.rename(columns={'F_1_d_returns': 'SP&500'}, inplace=True)

    # --- 2. Convert daily returns to cumulative returns ---
    cum_returns = pd.DataFrame((daily_mean[['SP&500']] + 1).cumprod())

    # --- 3. Plot cumulative returns ---
    cum_returns.plot()
    plt.title('Cumulative Returns Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(title_fontsize='13', fontsize='11')
    plt.show()

    # --- 4. Compute CAGR (Compound Annual Growth Rate) ---
    trading_days_per_year = 252
    number_of_years = len(daily_mean) / trading_days_per_year

    ending_value = cum_returns['SP&500'].iloc[-1]
    beginning_value = cum_returns['SP&500'].iloc[1]  # avoid 0 to prevent division issues

    ratio = ending_value / beginning_value
    cagr = round((ratio ** (1 / number_of_years) - 1) * 100, 2)

    print(f'CAGR: {cagr}%')

    # --- 5. Compute Sharpe Ratio ---
    avg_daily_return = daily_mean['SP&500'].mean() * trading_days_per_year
    std_daily_return = daily_mean['SP&500'].std() * (trading_days_per_year ** 0.5)

    sharpe_ratio = avg_daily_return / std_daily_return
    print(f'Sharpe Ratio: {round(sharpe_ratio, 2)}')

    # --- 6. Compute calendar year returns ---
    # Agrupar por año y calcular retornos acumulados por año
    ann_returns = (
        (daily_mean[['SP&500']] + 1)
        .groupby(daily_mean.index.get_level_values(0).year)
        .cumprod() - 1
    ) * 100

    # Tomar el último valor de cada año (retorno total anual)
    calendar_returns = pd.DataFrame(
        ann_returns['SP&500']
        .groupby(daily_mean.index.get_level_values(0).year)
        .last()
    )

    # --- 7. Plot calendar returns ---
    calendar_returns.plot.bar(rot=30, legend=False)
    plt.title("Calendar Year Returns", fontsize=14, fontweight='bold')
    plt.ylabel("Return (%)", fontsize=12)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    return cum_returns, calendar_returns


def calculate_rsi(returns, window=14):
    """
    Calcula el RSI (Relative Strength Index) de una serie de retornos.
    
    Parámetros:
    ------------
    returns : Series (pandas)
        Serie de retornos diarios o periódicos (pueden ser de precios, log returns, etc.)
    
    window : int (opcional, por defecto=14)
        Número de días (ventana) usado para calcular las medias móviles de ganancias y pérdidas.
    
    Retorna:
    --------
    rsi : Series
        Serie del RSI, con valores entre 0 y 100.
    """

    # --- Paso 1: Calcular las ganancias positivas en los retornos ---
    gain = returns[returns > 0].dropna().rolling(window=window).mean()
    gain.name = 'gain'  # Se asigna nombre a la serie para luego integrarla al DataFrame

    # --- Paso 2: Calcular las pérdidas (valores negativos de retorno) ---
    loss = returns[returns < 0].dropna().rolling(window=window).mean()
    loss.name = 'loss'

    # --- Paso 3: Añadir las columnas de gain y loss al DataFrame de retornos original ---
    returns = pd.merge(returns, gain, left_index=True, right_index=True, how='left')
    returns = pd.merge(returns, loss, left_index=True, right_index=True, how='left')

    # --- Paso 4: Rellenar valores faltantes hacia adelante (forward fill) ---
    # Esto es necesario porque el cálculo de medias móviles produce NaNs al principio
    returns = returns.ffill()

    # --- Paso 5: Eliminar cualquier fila que aún tenga valores nulos ---
    returns.dropna(inplace=True)

    # --- Paso 6: Calcular el "RS" (Relative Strength) ---
    # RS = media de ganancias / valor absoluto de la media de pérdidas
    ratio = returns['gain'] / abs(returns['loss'])

    # --- Paso 7: Calcular el RSI con la fórmula estándar ---
    # RSI = 100 - (100 / (1 + RS))
    rsi = 100 - (100 / (1 + ratio))

    return rsi

def compute_strat_perf(total_returns, cum_returns, calendar_returns, trading_strategy, model_name):
    """
    Evalúa el rendimiento de una estrategia de trading basada en una señal como RSI, MACD, etc.
    
    Parámetros:
    ------------
    total_returns : pd.DataFrame
        DataFrame que contiene los retornos diarios por activo y la columna con la señal del modelo.

    cum_returns : pd.DataFrame
        DataFrame que guarda los retornos acumulados (por cada estrategia o modelo).

    calendar_returns : pd.DataFrame
        DataFrame que guarda los retornos anuales (por cada estrategia o modelo).

    trading_strategy : función
        Función que aplica lógica de decisión (ej. comprar si RSI < 30, vender si RSI > 70).

    model_name : str
        Nombre de la columna del modelo que contiene la señal (por ejemplo: 'RSI').

    Retorna:
    --------
    cum_returns : pd.DataFrame
        Retornos acumulados actualizados con la estrategia aplicada.

    calendar_returns : pd.DataFrame
        Retornos anuales por año calendario, aplicando la estrategia.
    """

    # 1. Aplicar la estrategia de trading a cada valor de señal
    total_returns['Position'] = total_returns[model_name].transform(trading_strategy)

    # 2. Calcular el retorno diario condicionado a la posición tomada (0 = no invertir, 1 = invertir)
    total_returns[f'{model_name}_Return'] = total_returns['F_1_d_returns'] * total_returns['Position']

    # 3. Agrupar por fecha y calcular el retorno promedio diario (equal-weighted benchmark)
    daily_mean = pd.DataFrame(
        total_returns.loc[:, f'{model_name}_Return']
        .groupby(level='Date')
        .mean()
    )

    # 4. Calcular retornos acumulados
    cum_returns.loc[:, f'{model_name}_Return'] = ((daily_mean[[f'{model_name}_Return']] + 1).cumprod())

    # 5. Visualizar la evolución de los retornos acumulados
    cum_returns.plot()
    plt.title('Cumulative Returns Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(title_fontsize='13', fontsize='11')
    plt.show()

    # 6. Calcular CAGR (Compound Annual Growth Rate)
    number_of_years = len(daily_mean) / 252  # 252 días de trading al año
    ending_value = cum_returns[f'{model_name}_Return'].iloc[-1]
    beginning_value = cum_returns[f'{model_name}_Return'].iloc[1]
    ratio = ending_value / beginning_value
    cagr = round((ratio ** (1 / number_of_years) - 1) * 100, 2)
    print(f'The CAGR is: {cagr}%')

    # 7. Calcular Sharpe Ratio (retorno ajustado por riesgo)
    average_daily_return = daily_mean[[f'{model_name}_Return']].describe().iloc[1, :] * 252
    std_daily_return = daily_mean[[f'{model_name}_Return']].describe().iloc[2, :] * (252 ** 0.5)
    sharpe = average_daily_return / std_daily_return
    print(f'Sharpe Ratio of Strategy: {round(sharpe.iloc[0], 2)}')

    # 8. Calcular retornos anuales por calendario
    ann_returns = (
        ((daily_mean[f'{model_name}_Return'] + 1)
         .groupby(daily_mean.index.get_level_values(0).year)
         .cumprod()) - 1
    ) * 100

    calendar_returns.loc[:, f'{model_name}_Return'] = ann_returns.groupby(
        daily_mean.index.get_level_values(0).year
    ).last()

    # 9. Visualización de retornos anuales por año
    calendar_returns.plot.bar(rot=30, legend='top_left')
    plt.title('Calendar Year Returns')
    plt.ylabel('Return (%)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    return cum_returns, calendar_returns

