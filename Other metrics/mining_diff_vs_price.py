import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import seaborn as sns
import cryptocompare as cc
import requests
import yaml
import json

# Set the API Key from a yaml file
def get_api_key():
    yaml_file = open('C:/Users/maxim/Documents/projectESMEBITCOIN/bitcoin-explorer/env/api_config_cryptocompare.yml', 'r')
    p = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return p['api_key']

# Generic function for an API call to a given URL
def api_call(url):
    headers = {'authorization': 'Apikey ' + get_api_key()}
    session = requests.Session()
    session.headers.update(headers)
    response = session.get(url)
    historic_blockdata_dict = json.loads(response.text)
    df = pd.DataFrame.from_dict(historic_blockdata_dict.get('Data').get('Data'), orient='columns', dtype=None, columns=None)
    return df

# Prepare price data
def prepare_price_data(df):
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df.drop(columns=['time', 'conversionType', 'conversionSymbol'], inplace=True)
    return df

# Prepare on-chain data
def prepare_onchain_data(df):
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df.drop(columns='time', inplace=True)
    df = df[df['hashrate'] > 0.0]
    return df

# Add halving dates as vertical lines
def add_halving_dates(ax, df_x_dates, df_ax1_y):
    halving_dates = ['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', '2024-03-12', '2028-06-01']
    dates_list = [datetime.strptime(date, '%Y-%m-%d').date() for date in halving_dates]
    for i, datex in enumerate(dates_list):
        halving_ts = pd.Timestamp(datex)
        x_max = df_x_dates.max() + timedelta(days=365)
        x_min = df_x_dates.min() - timedelta(days=365)
        if (halving_ts < x_max) and (halving_ts > x_min):
            ax.axvline(x=datex, color='purple', linewidth=1, linestyle='dashed')
            ax.text(x=datex + timedelta(days=20), y=df_ax1_y.max()*0.99, s='BTC Halving ' + str(i) + '\n' + str(datex), color='purple')

# Number of past days for which we retrieve data
data_limit = 2000

# Define coin symbols
symbol_a = 'BTC'

# Load price data
base_url = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym='
df_a = api_call(f'{base_url}{symbol_a}&tsym=USD&limit={data_limit}')
coin_a_price_df = prepare_price_data(df_a)

# Load on-chain data
base_url = 'https://min-api.cryptocompare.com/data/blockchain/histo/day?fsym='
df_a = api_call(f'{base_url}{symbol_a}&limit={data_limit}')
onchain_symbol_a_df = prepare_onchain_data(df_a)

# Adding moving averages
rolling_window = 25
coin_a_price_df['close_avg'] = coin_a_price_df['close'].rolling(window=rolling_window).mean()

# Creating a Lineplot: Mining Difficulty vs Price
fig, ax1 = plt.subplots(figsize=(16, 6))
sns.lineplot(data=onchain_symbol_a_df, x='date', y='difficulty', linewidth=1.0, color='orangered', ax=ax1, label=f'mining difficulty')
ax2 = ax1.twinx()
sns.lineplot(data=coin_a_price_df, x='date', y='close', linewidth=0.5, color='skyblue', ax=ax2, label=f'close price')
sns.lineplot(data=coin_a_price_df, x='date', y='close_avg', linewidth=1.0, linestyle='--', color='royalblue', ax=ax2, label=f'MA-100')

add_halving_dates(ax1, onchain_symbol_a_df.date, onchain_symbol_a_df.hashrate)
ax1.set(xlabel=None)
ax1.set(ylabel='Mining Difficulty')
plt.title(f'{symbol_a} Mining Difficulty vs Price')
plt.show()














"""
A FAIRE CSV 
"""