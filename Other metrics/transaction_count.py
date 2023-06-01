"""
########################GRAPH LOG SCALE###################


import requests
import pandas as pd
import matplotlib.pyplot as plt
#from mplcursors import highlight, tooltip

# insert your API key here
API_KEY = "2OSWp0eKXPh5A4BV1TQN0tqZogg"

# make API request
res = requests.get('https://api.glassnode.com/v1/metrics/transactions/count',
    params={'a': 'BTC', 'api_key': API_KEY})

# convert to pandas dataframe
df = pd.read_json(res.text, convert_dates=['t'])

# plot the data
fig, ax = plt.subplots()
scatter = ax.scatter(df['t'], df['v'], s=5, color='navy')
#highlight(scatter, tooltip=tooltip(fmt='Date: %s\nTransaction Count: %s'))

# add the moving average line
df['average values'] = df['v'].rolling(window=50).mean()
df.plot(x='t', y='average values', color='gold', ax=ax, linewidth=1.5)

# set axis labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Transaction Count (log scale)')
ax.set_title("Bitcoin Transactions Count")

# enable zooming and add grid
ax.set(xlim=(df['t'].iloc[0], df['t'].iloc[-1]))
ax.grid(which='major', color='#CCCCCC', linestyle='--', linewidth=0.5)

# set color and linewidth of grid lines
ax.grid(which='major', linestyle='-', linewidth='0.5', color='lightgrey')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgrey')

# set log scale for y-axis
#ax.set_yscale('log')

plt.savefig('graph.png', dpi=400)

# show the plot
plt.show()



#####################GRAPH##################



import requests
import pandas as pd
import matplotlib.pyplot as plt

# insert your API key here
API_KEY = "2OSWp0eKXPh5A4BV1TQN0tqZogg"

# make API request
res = requests.get('https://api.glassnode.com/v1/metrics/transactions/count',
    params={'a': 'BTC', 'api_key': API_KEY})

# convert to pandas dataframe
df = pd.read_json(res.text, convert_dates=['t'])

# plot the data
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(df['t'], df['v'], s=5, color='navy')

# add the moving average line
df['average values'] = df['v'].rolling(window=50).mean()
df.plot(x='t', y='average values', color='gold', ax=ax, linewidth=1.5)

# set axis labels and title
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Transaction Count (log scale)', fontsize=14)
ax.set_title("Bitcoin Transactions Count", fontsize=20)

# enable zooming and add grid
ax.set(xlim=(df['t'].iloc[0], df['t'].iloc[-1]))
ax.grid(which='major', color='#CCCCCC', linestyle='--', linewidth=0.5)

# set color and linewidth of grid lines
ax.grid(which='major', linestyle='-', linewidth='0.5', color='lightgrey')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgrey')

# set log scale for y-axis
ax.set_yscale('log')

# set legend font size
ax.legend(loc='upper left', fontsize=14)

plt.show()

"""










# Tested with Python 3.9.13, Matplotlib 3.5.2, Seaborn 0.11.2, numpy 1.21.5, plotly 4.1.1, cryptocompare 0.7.6

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from datetime import date, timedelta, datetime
import seaborn as sns
sns.set_style('white', {'axes.spines.right': True, 'axes.spines.top': False})
import cryptocompare as cc
import requests
import IPython
import yaml
import json



# Set the API Key from a yaml file
yaml_file = open('C:/Users/maxim/Documents/projectESMEBITCOIN/bitcoin-explorer/env/api_config_cryptocompare.yml', 'r')  
p = yaml.load(yaml_file, Loader=yaml.FullLoader)
api_key = p['api_key'] 

# Number of past days for which we retrieve data
data_limit = 2000

# Define coin symbols
symbol_a = 'BTC'



# Query price data

# Generic function for an API call to a given URL
def api_call(url):
    # Set API Key as Header
    headers = {'authorization': 'Apikey ' + api_key,}
    session = requests.Session()
    session.headers.update(headers)

    # API call to cryptocompare
    response = session.get(url)

    # Conversion of the response to dataframe
    historic_blockdata_dict = json.loads(response.text)
    df = pd.DataFrame.from_dict(historic_blockdata_dict.get('Data').get('Data'), orient='columns', dtype=None, columns=None)
    return df

def prepare_pricedata(df):
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df.drop(columns=['time', 'conversionType', 'conversionSymbol'], inplace=True)
    return df

# Load the price data
base_url = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym='
df_a = api_call(f'{base_url}{symbol_a}&tsym=USD&limit={data_limit}')
coin_a_price_df = prepare_pricedata(df_a)






# Query on-chain data

# Prepare the onchain dataframe
def prepare_onchain_data(df):
  # replace the timestamp with a data and filter some faulty values
  df['date'] = pd.to_datetime(df['time'], unit='s')
  df.drop(columns='time', inplace=True)
  df = df[df['hashrate'] > 0.0]
  return df
  
base_url = 'https://min-api.cryptocompare.com/data/blockchain/histo/day?fsym='
onchain_symbol_a_df = api_call(f'{base_url}{symbol_a}&limit={data_limit}')


# Filter some faulty values
onchain_symbol_a_df = onchain_symbol_a_df[onchain_symbol_a_df['hashrate'] > 0.0]
onchain_symbol_a_df.head(3)





# Lineplot: Price Correlation with Bitcoin

# Adding moving averages
rolling_window = 25
coin_a_price_df['close_avg'] = coin_a_price_df['close'].rolling(window=rolling_window).mean() 


# This function adds bitcoin halving dates as vertical lines
def add_halving_dates(ax, df_x_dates, df_ax1_y):
    halving_dates = ['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', '2024-03-12', '2028-06-01'] 
    dates_list = [datetime.strptime(date, '%Y-%m-%d').date() for date in halving_dates]
    for i, datex in enumerate(dates_list):
        halving_ts = pd.Timestamp(datex)
        x_max = df_x_dates.max() + timedelta(days=365)
        x_min = df_x_dates.min() - timedelta(days=365)
        if (halving_ts < x_max) and (halving_ts > x_min):
            ax.axvline(x=datex, color = 'purple', linewidth=1, linestyle='dashed')
            ax.text(x=datex  + timedelta(days=20), y=df_ax1_y.max()*0.99, s='BTC Halving ' + str(i) + '\n' + str(datex), color = 'purple')






# Create the lineplot
fig, ax1 = plt.subplots(figsize=(16, 6))
sns.lineplot(data=coin_a_price_df, x='date', y='close', color='cornflowerblue', linewidth=0.5, label=f'{symbol_a} close price', ax=ax1)
sns.lineplot(data=coin_a_price_df, x='date', y='close_avg', color='blue', linestyle='dashed', linewidth=1.0, 
    label=f'{symbol_a} {rolling_window}-MA', ax=ax1)
ax1.set_ylabel(f'{symbol_a} Prices')
ax1.set(xlabel=None)
ax2 = ax1.twinx()
add_halving_dates(ax1, coin_a_price_df.date, coin_a_price_df.close)
plt.show()




# Prepare the onchain dataframe
def prepare_onchain_data(df):
  # replace the timestamp with a data and filter some faulty values
  df['date'] = pd.to_datetime(df['time'], unit='s')
  df.drop(columns='time', inplace=True)
  df = df[df['hashrate'] > 0.0]
  return df

# Load onchain data for Bitcoin
base_url = 'https://min-api.cryptocompare.com/data/blockchain/histo/day?fsym='
df_a = api_call(f'{base_url}{symbol_a}&limit={data_limit}')
onchain_symbol_a_df = prepare_onchain_data(df_a)






# Prepare balance distribution dataframe
def prepare_balancedistribution_data(df):
  df['balance_distribution'] = df['balance_distribution'].apply(lambda x: [i for i in x])
  json_struct = json.loads(df[['time','balance_distribution']].to_json(orient="records"))    
  df_ = pd.json_normalize(json_struct)
  df_['date'] = pd.to_datetime(df_['time'], unit='s')
  df_flat = pd.concat([df_.explode('balance_distribution').drop(['balance_distribution'], axis=1),
           df_.explode('balance_distribution')['balance_distribution'].apply(pd.Series)], axis=1)
  df_flat.reset_index(drop=True, inplace=True)
  df_flat['range'] = ['' + str(float(df_flat['from'][x])) + '_to_' + str(float(df_flat['to'][x])) for x in range(df_flat.shape[0])]
  df_flat.drop(columns=['from','to', 'time'], inplace=True)

  # Data cleansing
  df_flat = df_flat[~df_flat['range'].isin(['100000.0_to_0.0'])]
  df_flat['range'].iloc[df_flat['range'] == '1e-08_to_0.001'] = '0.0_to_0.001'
  return df_flat

# Load the balance distribution data for Bitcoin
base_url = 'https://min-api.cryptocompare.com/data/blockchain/balancedistribution/histo/day?fsym='
df_raw = api_call(f'{base_url}{symbol_a}&limit={data_limit}')
df_distr = prepare_balancedistribution_data(df_raw)
df_distr.head(3)





# Prepare address distribution data for plotting
df_distr_add = df_distr.copy()
for i in list(df_distr_add.range.unique()):
    df_distr_add.loc[df_distr.range == i, 'addressesCount_pct_cum'] = df_distr_add[df_distr_add.range == i]['addressesCount'].pct_change().dropna().cumsum().rolling(window=50).mean()
df_distr_add.dropna(inplace=True)

# Lineplot: Address Count by Holder Amount
fig, ax1 = plt.subplots(figsize=(16, 6))
sns.lineplot(data=df_distr_add, x='date', hue='range', linewidth = 1.0, y='addressesCount_pct_cum', ax=ax1, palette='bright')
plt.ylabel('Percentage Growth')
ax1.tick_params(axis="x", rotation=90, labelsize=10, length=0)
ax1.set(xlabel=None)
plt.title(f'Percentage Growth in the Distribution of Total Address Count for {symbol_a} by Holder Amount')
plt.legend(framealpha=0)
plt.show()




# Lineplot: Difficulty vs Hashrate
fig, ax1 = plt.subplots(figsize=(16, 6))
sns.lineplot(data=onchain_symbol_a_df, x='date', y='difficulty', 
    linewidth=1.0, color='royalblue', ax=ax1, label=f'{symbol_a} mining difficulty')
ax2 = ax1.twinx()
sns.lineplot(data=onchain_symbol_a_df[::5], x='date', y='hashrate', 
    linewidth=1.0, color='red', ax=ax2, label=f'{symbol_a} network hashrate')

add_halving_dates(ax1, onchain_symbol_a_df.date, onchain_symbol_a_df.difficulty)
ax1.set(xlabel=None)
plt.title(f'{symbol_a} Mining Difficulty vs Hashrate')
plt.show()



# Add a moving average
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




# Calculate active addresses moving average
rolling_window=25
y_a_add_ma = onchain_symbol_a_df['active_addresses'].rolling(window=rolling_window).mean() 


# Lineplot: Active Addresses
fig, ax1 = plt.subplots(figsize=(16, 6))
sns.lineplot(data=onchain_symbol_a_df[-1*data_limit::10], x='date', y='active_addresses', 
    linewidth=0.5, color='skyblue', ax=ax1, label=f'{symbol_a} active addresses')
sns.lineplot(data=onchain_symbol_a_df[-1*data_limit::10], x='date', y=y_a_add_ma, 
    linewidth=1.0, color='royalblue', linestyle='--', ax=ax1, label=f'{symbol_a} active addresses {rolling_window}-Day MA')

ax1.set(xlabel=None)
ax1.set(ylabel='Active Addresses')

plt.legend(framealpha=0)
plt.show()





# Calculate Transaction Count Moving Averages
rolling_window=25
y_a_trx_ma = onchain_symbol_a_df['transaction_count'].rolling(window=rolling_window).mean() 


# Lineplot: Transactions Count
fig, ax1 = plt.subplots(figsize=(16, 6))
sns.lineplot(data=onchain_symbol_a_df[-1*data_limit::10], x='date', y='transaction_count', 
    linewidth=0.5, color='skyblue', ax=ax1, label=f'{symbol_a} transactions')
sns.lineplot(data=onchain_symbol_a_df[-1*data_limit::10], x='date', y=y_a_trx_ma, 
    linewidth=1.0, color='royalblue', linestyle='--', ax=ax1, label=f'{symbol_a} transactions {rolling_window}-Day MA')

ax1.set(xlabel=None)
ax1.set(ylabel='Transaction Count')
plt.legend(framealpha=0)
plt.show()




# Calculate Large Transactions Moving Averages
rolling_window=25
y_a_ltrx_ma = onchain_symbol_a_df['large_transaction_count'].rolling(window=rolling_window).mean() 
 

# Lineplot: Large Transactions
fig, ax1 = plt.subplots(figsize=(16, 6))
sns.lineplot(data=onchain_symbol_a_df[-1*data_limit::10], x='date', y='large_transaction_count', 
    linewidth=0.5, color='skyblue', ax=ax1, label=f'{symbol_a} large transactions')
sns.lineplot(data=onchain_symbol_a_df[-1*data_limit::10], x='date', y=y_a_ltrx_ma, 
    linewidth=1.0, color='royalblue', linestyle='--', ax=ax1, label=f'{symbol_a} large transactions {rolling_window}-Day MA')
ax1.set(ylabel='Large Transactions')
plt.legend(framealpha=0)
plt.show()




