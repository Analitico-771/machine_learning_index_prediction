
# Import libraries and dependencies
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv


def alpaca_symbol_data(tickers, years_back):

    # Load the environment variables from the .env file
    # by calling the load_dotenv function
    load_dotenv()

    api = REST(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
        api_version = "v2"
    )

    # create the user_start_date by getting today's date
    user_start_date = date.today()
    # calculate yesterday's date to get yesterday's close price
    # yesterday's close price is the end_date
    yesterday = user_start_date - timedelta(days=1)
    # create the formatted_start_date to include the years back to get the data
    formatted_start_date = user_start_date.replace(year=(yesterday.year - years_back), month=yesterday.month, day=yesterday.day)
    # convert formatted_start_date to string for pd.Timestamp()
    formatted_start_date = str(formatted_start_date)
    # convert yesterday to string for pd.Timestamp()
    yesterday = str(yesterday)
    # convert to iso format
    start_date = pd.Timestamp(formatted_start_date, tz="America/New_York").isoformat()
    end_date = pd.Timestamp(yesterday, tz="America/New_York").isoformat()
    # Set limit_rows to 1000 to retreive the maximum amount of rows
    # limit_rows = 10000
        
    # Get 3 years worth of historical data for tickers
    ticker_data = api.get_bars(
        tickers,
        TimeFrame.Day,
        start=start_date,
        end=end_date,
        # limit=limit_rows
    ).df

    # print(formatted_date)
    # print(yesterday)
    # print(start_date)
    # print(end_date)

    # Drop all empty/null/na cells
    ticker_data = ticker_data.dropna()

    # return df to main code
    return ticker_data


def yahoo_symbol_data(tickers, years_back):

    combined_tickers_list = []

    # create the user_start_date by getting today's date
    user_start_date = date.today()
    # calculate yesterday's date to get yesterday's close price
    # yesterday's close price is the end_date
    yesterday = user_start_date - timedelta(days=1)
    # create the formatted_start_date to include the years_back to get the data
    formatted_start_date = user_start_date.replace(year=(yesterday.year - years_back), month=yesterday.month, day=yesterday.day)
    # convert formatted_start_date to string for pd.Timestamp()
    formatted_start_date = str(formatted_start_date)
    # convert yesterday to string for pd.Timestamp()
    yesterday = str(yesterday)

    try:
        for each_ticker in tickers:
            data = yf.download(
                each_ticker,
                start=formatted_start_date,
                end=yesterday
            )
            # dropna() from df
            data = data.dropna()
            data = data.drop(columns=['Open','High','Low','Close','Volume'])
            # Rename column to ticker name
            data = data.rename(columns = {'Adj Close' : 'close'})
            # Append each ticker to a list
            combined_tickers_list.append(data)

    except Exception as error:
        print(error)

    # Concatenate the dataframes
    tickers_df = pd.concat(combined_tickers_list, axis="columns", join="inner")

    return tickers_df