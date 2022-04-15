import pandas as pd

def TalltoWide(tallDF):
    SPY = tallDF.loc[tallDF['symbol'] == 'SPY']
    SPY.drop(columns={'high','low','volume','trade_count','vwap','symbol'}, inplace=True)
    SPY.rename(columns={'open':'SPY Open', 'close':'SPY Close'}, inplace=True)
    tallDF['daily return'] = (tallDF['close'] - tallDF['open'])/tallDF['open']
    tallDF.drop(columns={'open','high','low','close','volume','trade_count','vwap'}, inplace=True)
    StocksWide = pd.pivot_table(tallDF, columns='symbol', index = 'timestamp', values='daily return')
    StocksWide.reset_index(inplace=True)
    StocksWide['SPY Open'] = SPY['SPY Open']
    StocksWide['SPY Close'] = SPY['SPY Close']
    return StocksWide


