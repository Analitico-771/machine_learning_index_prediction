import pandas as pd

def prep_data(WideDF, sectors, df500sectors):
    WideDF.drop(columns='index', inplace=True)
    WideDF = WideDF.set_index('timestamp')
    SectorDF = WideDF[['SPY', 'SPY Open', 'SPY Close']]
    stockstodrop = []
    StocksWideNoSPY = WideDF.drop(columns={'SPY','SPY Open','SPY Close'})
    for sector in sectors:
        for (columnName, columnValues) in StocksWideNoSPY.iteritems():
            if sector != df500sectors.loc[columnName]['GICS Sector']:
                stockstodrop.append(columnName)
        TempDF = StocksWideNoSPY.drop(columns=stockstodrop)
        stockstodrop = []
        SectorDF[sector] = TempDF.mean(axis=1)
    return SectorDF

