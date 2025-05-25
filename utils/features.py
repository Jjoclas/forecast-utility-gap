import pytz
from typing import List
import pandas as pd
import numpy as np
from talib import abstract as ta
from sklearn.model_selection import train_test_split


def split_dataset(dataset: pd.DataFrame, target_column: str, train_size: float = 0.7):
    """Split dataset into train, validation and test sets"""
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, shuffle=False
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def close_standardize(col: pd.Series, close: pd.Series, window: int = 100, min_std: float = 1e-8) -> pd.Series:
    """ Standardize a column with respect to the closing price
    args:
        col: pd.Series
        close: pd.Series
        window: int
        min_std: float
    return:
        pd
    """
    rolling_std = col.rolling(window=window).std()
    rolling_std[rolling_std < min_std] = min_std  # Set a minimum value for standard deviation
    return (col - close) / rolling_std


def talib_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Talib features
    args:
        df: pd.DataFrame
    return:
        df: pd.DataFrame
    """
    df["std"] = df['close'].rolling(window=100).std()
    df['date'] = pd.to_datetime(df['from'], unit='s', utc=True).dt.tz_convert(pytz.UTC)
    df.set_index(['date'], inplace=True)

    # Extracting date-related features
    df['HourOfDay'] = df.index.get_level_values('date').hour  # Hour of the day
    df['DayOfWeek'] = df.index.get_level_values('date').weekday  # Day of the week
    df['DayOfMonth'] = df.index.get_level_values('date').day  # Day of the month

    #Overlapping Studies

    df['Sma5'] = ta.SMA(df['close'], timeperiod=5) 
    df['Sma10'] = ta.SMA(df['close'], timeperiod=10) 
    df['UpperBB'], df['MiddleBB'], df['LowerBB'] = ta.BBANDS(df['close'], timeperiod=14)
    df['UpperBB'] = df['UpperBB']
    df['MiddleBB'] = df['MiddleBB']
    df['LowerBB'] = df['LowerBB']
    df['Sma5'] = close_standardize(df['Sma5'], df['close'], window=5)
    df['Sma10'] = close_standardize(df['Sma10'], df['close'], window=10)
    df['UpperBB'] = close_standardize(df['UpperBB'], df['close'], window=14)
    df['MiddleBB'] = close_standardize(df['MiddleBB'], df['close'], window=14)
    df['LowerBB'] = close_standardize(df['LowerBB'], df['close'], window=14)
    

    #Momentum Indicators
    df['AARON'] = ta.AROONOSC(df['max'], df['min'], timeperiod=10) / 100
    df['ADX'] = ta.ADX(df['max'], df['min'], df['close'], timeperiod=10) / 100
    df['MACD'], df['signal'], df['hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9) #normalized = True
    df['RSI'] = ta.RSI(df['close'], timeperiod=14) / 100
    df['ROC'] = ta.ROC(df['close'], timeperiod=10)
    df['MFM'] = ta.MFI(df['max'], df['min'], df['close'], df['volume'], timeperiod=10) / 100
    df['DX'] = ta.DX(df['max'], df['min'], df['close'], timeperiod=10) / 100
    df['MINUS_DI'] = ta.MINUS_DI(df['max'], df['min'], df['close'], timeperiod=10) / 100
    df['PLUS_DI'] = ta.PLUS_DI(df['max'], df['min'], df['close'], timeperiod=10) / 100
    df['MINUS_DM'] = ta.MINUS_DM(df['max'], df['min'], timeperiod=10) / 100
    df['PLUS_DM'] = ta.PLUS_DM(df['max'], df['min'], timeperiod=10) / 100
    df['STOCHF'] = ta.STOCHF(df['max'], df['min'], df['close'], fastk_period=10, fastd_period=10, fastd_matype=0)[0] / 100
    df['STOCH'] = ta.STOCH(df['max'], df['min'], df['close'], fastk_period=10, slowk_period=10, slowk_matype=0, slowd_period=10, slowd_matype=0)[0] / 100
    df['STOCHRSI'] = ta.STOCHRSI(df['close'], timeperiod=10, fastk_period=10, fastd_period=10, fastd_matype=0)[0] / 100
    df['TRIX'] = ta.TRIX(df['close'], timeperiod=10)
    df['ULTOSC'] = ta.ULTOSC(df['max'], df['min'], df['close'], timeperiod1=10, timeperiod2=10, timeperiod3=10) / 100
    df['WILLR'] = ta.WILLR(df['max'], df['min'], df['close'], timeperiod=10) / 100
    #Volatility Indicators
    df['ATR'] = ta.ATR(df['max'], df['min'], df['close'], timeperiod=10)
    df['NATR'] = ta.NATR(df['max'], df['min'], df['close'], timeperiod=10) / 100
    df['TRANGE'] = ta.TRANGE(df['max'], df['min'], df['close'])

    #Volumne Indicators
    df['OBV'] = ta.OBV(df['close'], df['volume'])
    df['OBV'] = (df['OBV'] - df['OBV'].mean()) / df['OBV'].rolling(window=200).std()
    df['AD'] = ta.AD(df['max'], df['min'], df['close'], df['volume'])
    df['AD'] = (df['AD'] - df['AD'].mean()) / df['AD'].rolling(window=200).std()
    
    #Cycle Indicators
    df['HT_TRENDLINE'] = ta.HT_TRENDLINE(df['close'])
    df['HT_TRENDLINE'] = close_standardize(df['HT_TRENDLINE'], df['close'], window=100)

    #Statistic Functions
    df['TSF'] = ta.TSF(df['close'], timeperiod=10)
    df['TSF'] = close_standardize(df['TSF'], df['close'], window=100)
    df['VAR'] = ta.VAR(df['close'], timeperiod=10)

    #Pattern Recognition
    df['CDL3INSIDE'] = ta.CDL3INSIDE(df['open'], df['max'], df['min'], df['close'])
    df['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(df['open'], df['max'], df['min'], df['close'])
    df['CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(df['open'], df['max'], df['min'], df['close'])
    df['CDLBELTHOLD'] = ta.CDLBELTHOLD(df['open'], df['max'], df['min'], df['close'])
    df['CDLCLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(df['open'], df['max'], df['min'], df['close'])
    df['CDLDOJI'] = ta.CDLDOJI(df['open'], df['max'], df['min'], df['close'])
    df['CDLDOJISTAR'] = ta.CDLDOJISTAR(df['open'], df['max'], df['min'], df['close'])
    df['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(df['open'], df['max'], df['min'], df['close'])
    df['CDLENGULFING'] = ta.CDLENGULFING(df['open'], df['max'], df['min'], df['close'])
    df['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df['open'], df['max'], df['min'], df['close'])
    df['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(df['open'], df['max'], df['min'], df['close'])
    df['CDLHAMMER'] = ta.CDLHAMMER(df['open'], df['max'], df['min'], df['close'])
    df['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df['open'], df['max'], df['min'], df['close'])
    df['CDLHARAMI'] = ta.CDLHARAMI(df['open'], df['max'], df['min'], df['close'])
    df['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(df['open'], df['max'], df['min'], df['close'])
    df['CDLHIGHWAVE'] = ta.CDLHIGHWAVE(df['open'], df['max'], df['min'], df['close'])
    df['CDLHIKKAKE'] = ta.CDLHIKKAKE(df['open'], df['max'], df['min'], df['close'])
    df['CDLIDENTICAL3CROWS'] = ta.CDLIDENTICAL3CROWS(df['open'], df['max'], df['min'], df['close'])
    df['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df['open'], df['max'], df['min'], df['close'])
    df['CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(df['open'], df['max'], df['min'], df['close'])
    df['CDLLONGLINE'] = ta.CDLLONGLINE(df['open'], df['max'], df['min'], df['close'])
    df['CDLMARUBOZU'] = ta.CDLMARUBOZU(df['open'], df['max'], df['min'], df['close'])
    df['CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(df['open'], df['max'], df['min'], df['close'])
    df['CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(df['open'], df['max'], df['min'], df['close'])
    df['CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(df['open'], df['max'], df['min'], df['close'])
    df['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df['open'], df['max'], df['min'], df['close'])
    df['CDLSHORTLINE'] = ta.CDLSHORTLINE(df['open'], df['max'], df['min'], df['close'])
    df['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df['open'], df['max'], df['min'], df['close'])
    df['CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(df['open'], df['max'], df['min'], df['close'])
    df['CDLTAKURI'] = ta.CDLTAKURI(df['open'], df['max'], df['min'], df['close'])
    df['CDLTHRUSTING'] = ta.CDLTHRUSTING(df['open'], df['max'], df['min'], df['close'])
    df['CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(df['open'], df['max'], df['min'], df['close'])    

    #Price Transform
    df['Ewm200'] = ta.EMA(df['close'], timeperiod=200)
    df['Ewm100'] = ta.EMA(df['close'], timeperiod=100)
    df['Ewm50'] = ta.EMA(df['close'], timeperiod=50) 
    df['Ewm10'] = ta.EMA(df['close'], timeperiod=10)
    df['Ewm5'] = ta.EMA(df['close'], timeperiod=5)
    df['Ewm200'] = close_standardize(df['Ewm200'], df['close'], window=200)
    df['Ewm100'] = close_standardize(df['Ewm100'], df['close'], window=100)
    df['Ewm50'] = close_standardize(df['Ewm50'], df['close'], window=50)
    df['Ewm10'] = close_standardize(df['Ewm10'], df['close'], window=10)
    df['Ewm5'] = close_standardize(df['Ewm5'], df['close'], window=5)

    df['diff_vela'] = df['close'] - df['open']
    df['avg_candle'] = df['diff_vela'].abs().rolling(window=100).mean()

    df['diff'] = df['close'].diff(-1).fillna(0) * -1
    df['x'] = (df['diff_vela'] ) / df['diff_vela'].abs().rolling(100).std()

    df['Y'] = df['x'].shift(-1)
    df['Y'] = df['Y'] / df['Y'].abs()

    return df.dropna()
