def calculate_rsi(series, length):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic_oscillator(series, k_period, d_period):
    low_min = series.rolling(window=k_period).min()
    high_max = series.rolling(window=k_period).max()
    k = 100 * ((series - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

