
import pandas as pd 
import matplotlib.pyplot as plt  
#from datetime import date, timedelta, datetime
import seaborn as sns
import requests
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

# Query on-chain data

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

# Filter some faulty values
onchain_symbol_a_df = onchain_symbol_a_df[onchain_symbol_a_df['hashrate'] > 0.0]
onchain_symbol_a_df.head(3)

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


