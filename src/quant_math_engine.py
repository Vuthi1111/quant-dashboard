import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def calculate_ou_parameters(returns: pd.Series, dt: float = 1.0):
    """
    Fits an Ornstein-Uhlenbeck process to the given return series.
    dX_t = \theta (\mu - X_t) dt + \sigma dW_t
    Returns (theta, mu, sigma, half_life).
    """
    if len(returns) < 10:
        return np.nan, np.nan, np.nan, np.nan
        
    # We estimate via OLS regression: X_{t} - X_{t-1} = a + b * X_{t-1} + e
    y = returns.diff().dropna()
    x = returns.shift(1).dropna()
    
    # Align indices
    common_idx = y.index.intersection(x.index)
    if len(common_idx) < 10:
        return np.nan, np.nan, np.nan, np.nan
        
    y = y.loc[common_idx]
    x = sm.add_constant(x.loc[common_idx])
    
    model = sm.OLS(y, x).fit()
    
    if len(model.params) < 2:
        return np.nan, np.nan, np.nan, np.nan
        
    a, b = model.params
    
    # \theta = -b / dt
    theta = -b / dt
    
    # \mu = a / -b
    mu = a / -b if b != 0 else np.nan
    
    # \sigma = std(e) / sqrt(dt)
    sigma = np.std(model.resid) / np.sqrt(dt)
    
    # Half-life = ln(2) / \theta
    half_life = np.log(2) / theta if theta > 0 else np.nan
    
    return theta, mu, sigma, half_life

def calculate_hurst(ts: pd.Series, min_window=10):
    """
    Calculates the Hurst Exponent using Detrended Fluctuation Analysis (DFA) proxy.
    H < 0.5: Mean Reverting
    H = 0.5: Random Walk
    H > 0.5: Trending
    """
    if len(ts) < min_window * 2:
        return np.nan
        
    lags = range(2, min_window)
    tau = [np.sqrt(np.std(np.subtract(ts.values[lag:], ts.values[:-lag]))) for lag in lags]
    
    # Fit a line to log-log plot
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    return poly[0] * 2.0

def calculate_dynamic_fair_value(gold_prices: pd.Series, dxy_prices: pd.Series, tips_yields: pd.Series, window=252):
    """
    Uses rolling OLS (proxy for Kalman Filter dynamic beta) to compute the Fair Value Gap (Z-Score)
    """
    df = pd.concat([gold_prices, dxy_prices, tips_yields], axis=1).dropna()
    df.columns = ['Gold', 'DXY', 'TIPS']
    
    if df.empty:
        return pd.Series(index=gold_prices.index, dtype=float)
        
    df['Gold_Log'] = np.log(df['Gold'])
    df['DXY_Log'] = np.log(df['DXY'])
    
    z_scores = pd.Series(index=df.index, dtype=float, name="coint_z_score")
    coint_fvs = pd.Series(index=df.index, dtype=float, name="coint_fv")
    coint_stds = pd.Series(index=df.index, dtype=float, name="coint_std")
    
    # Rolling regression to calculate the spread (error correction term)
    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i]
        
        y = window_df['Gold_Log']
        X = sm.add_constant(window_df[['DXY_Log', 'TIPS']])
        
        try:
            model = sm.OLS(y, X).fit()
            
            # Predict the current Fair Value based on the rolling betas
            current_X = np.array([1, df['DXY_Log'].iloc[i], df['TIPS'].iloc[i]])
            fv = np.dot(model.params, current_X)
            
            # Calculate residual (Error)
            residual = df['Gold_Log'].iloc[i] - fv
            
            # Z-Score of the residual based on historical rolling std
            std_resid = np.std(model.resid)
            z = residual / std_resid if std_resid > 0 else 0
            
            z_scores.iloc[i] = z
            coint_fvs.iloc[i] = fv
            coint_stds.iloc[i] = std_resid
        except:
            z_scores.iloc[i] = np.nan
            coint_fvs.iloc[i] = np.nan
            coint_stds.iloc[i] = np.nan
            
    return pd.concat([z_scores, coint_fvs, coint_stds], axis=1)
