import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import requests
import IPython
import yaml
import json
import seaborn as sns
# Set the API Key from a yaml file
yaml_file = open('C:/Users/maxim/Documents/projectESMEBITCOIN/bitcoin-explorer/env/api_config_cryptocompare.yml', 'r')
p = yaml.load(yaml_file, Loader=yaml.FullLoader)
api_key = p['api_key']

# Number of past days for which we retrieve data
data_limit = 2000

# Define coin symbols
symbol_a = 'BTC'

# Query on-chain data
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
# Prepare the onchain dataframe
def prepare_onchain_data(df):
    # replace the timestamp with a date and filter some faulty values
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df.drop(columns='time', inplace=True)
    df = df[df['hashrate'] > 0.0]
    return df


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
sns.lineplot(data=df_distr_add, x='date', hue='range', linewidth=1.0, y='addressesCount_pct_cum', ax=ax1, palette='bright')
plt.ylabel('Percentage Growth')
ax1.tick_params(axis="x", rotation=90, labelsize=10, length=0)
ax1.set(xlabel=None)
plt.title(f'Percentage Growth in the Distribution of Total Address Count for {symbol_a} by Holder Amount')
plt.legend(framealpha=0)

# Ajouter le quadrillage
ax1.grid(True)

plt.show()
"""
# Save the plot data to CSV
plot_data = df_distr_add[['date', 'range', 'addressesCount_pct_cum']]
plot_data.to_csv('distribution_plot_data.csv', index=False)

"""