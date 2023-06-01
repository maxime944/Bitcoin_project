from NODE_DIFFICULTY import combined_curve_normalized
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(12, 6))
plt.plot(combined_curve_normalized.index, combined_curve_normalized.values)
plt.xlabel('Index')
plt.ylabel('Valeurs normalisées')
plt.title('Série chronologique normalisée')
plt.grid(True)
plt.show()

print(combined_curve_normalized)

# Convertir la série chronologique en DataFrame
df = pd.DataFrame({'values': combined_curve_normalized})

# Créer un index numérique à partir de 0 jusqu'à la longueur de la série chronologique
df['year'] = np.arange(2009, 2009 + len(df))

# Utiliser une interpolation linéaire pour générer des valeurs supplémentaires avec un pas de 1/12 entre les mois
new_years = np.arange(2009, 2009 + len(df) + 1/12, 1/12)
interpolated_values = np.interp(new_years, df['year'], df['values'])

# Créer une nouvelle série chronologique avec les valeurs interpolées et les nouvelles années
interpolated_series = pd.Series(interpolated_values, index=new_years)

# Formater l'affichage des décimales
pd.set_option('display.float_format', '{:.6f}'.format)

# Créer un nouvel index de dates
start_date = pd.to_datetime('2009-01-01')
end_date = start_date + pd.DateOffset(months=len(interpolated_series)-1)
new_index = pd.date_range(start=start_date, end=end_date, freq='MS')

# Réindexer la série chronologique avec le nouvel index de dates
interpolated_series.index = new_index

# Afficher la nouvelle série chronologique
pd.set_option('display.max_rows', None)
#print(interpolated_series)


# Supprimer les 8 dernières lignes de la série interpolée
interpolated_series = interpolated_series[:-12]

# Afficher la série chronologique mise à jour
print(interpolated_series)


plt.plot(interpolated_series)
plt.xlabel('Date')
plt.ylabel('Valeur')
plt.title('Interpolated Series')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



###################ADFULLER######################

from statsmodels.tsa.stattools import adfuller

# Appliquer le test ADF
result = adfuller(interpolated_series)

# Extraire les résultats du test
adf_statistic = result[0]
p_value = result[1]
critical_values = result[4]

# Afficher les résultats
print("Résultats du test ADF:")
print(f"Statistique ADF : {adf_statistic}")
print(f"p-valeur : {p_value:.2f}")
print("Valeurs critiques :")
for key, value in critical_values.items():
    print(f"    {key}: {value}")
    
    
    
"""
Interprétation des résultats du test ADF :

Statistique ADF : -295383936016839.25
La statistique ADF est une mesure de la présence de racines unitaires dans la série temporelle. En général, plus la valeur absolue de la statistique est négative et plus elle s'éloigne de zéro, plus il est probable que la série temporelle soit stationnaire. Dans ce cas, la statistique ADF est extrêmement négative, ce qui suggère une forte probabilité de stationnarité.

p-valeur : 0.0
La p-valeur est utilisée pour prendre une décision quant à l'hypothèse nulle du test ADF. En règle générale, si la p-valeur est inférieure à un seuil spécifique (généralement 0,05), on rejette l'hypothèse nulle et on conclut que la série temporelle est stationnaire. Dans ce cas, la p-valeur est de 0.0, ce qui indique une forte probabilité de stationnarité.

Valeurs critiques :
Les valeurs critiques correspondent aux seuils au-delà desquels nous pouvons rejeter l'hypothèse nulle. Les valeurs critiques sont spécifiques aux différents niveaux de signification (1%, 5%, 10%) utilisés dans le test ADF. Dans ce cas, la statistique ADF est inférieure à toutes les valeurs critiques, ce qui renforce l'idée de stationnarité de la série temporelle.

En conclusion, les résultats du test ADF suggèrent fortement que votre série temporelle est stationnaire, ce qui est un bon point de départ pour l'application d'un modèle ARIMA.
"""


##########FIGURE OUT ORDER FOR ARIMA MODEL################

from pmdarima import auto_arima
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore")

# Ajuster auto_arima à votre série de données
stepwise_fit = auto_arima(interpolated_series, trace=True, suppress_warnings=True)

# Afficher le résumé
stepwise_fit.summary()


#############ARIMA MODEL START################


from statsmodels.tsa.arima.model import ARIMA

print(interpolated_series.shape)

train = interpolated_series.iloc[:-50]
test = interpolated_series.iloc[-50:]

print(train.shape, test.shape)


#############TRAIN MODEL ARIMA############


model = ARIMA(interpolated_series.values, order=(0, 2, 0))
model_fit = model.fit()
print(model_fit.summary())


###########MAKE PRED ON TEST SET################
# Obtain the estimated parameters
params = model_fit.params

# Make predictions using the fitted model
start = len(train)
end = len(train) + len(test) - 1
pred = model_fit.predict(start=start, end=end, typ='levels', params=params)
#print(pred)


pred_index = pd.RangeIndex(start, end+1)
pred = pd.Series(pred, index=pred_index)
print(pred)
start_date = pd.to_datetime('2009-01-01')



##############CHECK DATES######################

date_168 = start_date + pd.DateOffset(months=168)
print(date_168)

date_119 = start_date + pd.DateOffset(months=119)
print(date_119)


##############################################

import pandas as pd

# Créer la série de données
data = pd.Series([0.195898, 0.191139, 0.186381, 0.182290, 0.177863, 0.173437, 0.169012, 0.164587, 0.160161, 0.155736, 0.151310, 0.146885, 0.142459, 0.138034, 0.133609, 0.129805, 0.125687, 0.121571, 0.117456, 0.113340, 0.109224, 0.105109, 0.100993, 0.096877, 0.092762, 0.088646, 0.084531, 0.080993, 0.077163, 0.073336, 0.069508, 0.065681, 0.061853, 0.058026, 0.054198, 0.050371, 0.046543, 0.042716, 0.038888, 0.035598, 0.032037, 0.028477, 0.024917, 0.021358, 0.017798, 0.014239, 0.010679, 0.007119, 0.003560, 0.000000])

# Créer une liste de dates à partir de décembre 2018 jusqu'à janvier 2023
start_date = pd.to_datetime("2018-12-01")
end_date = pd.to_datetime("2023-01-01")
date_range = pd.date_range(start_date, end_date, freq='MS')

# Remplacer chaque valeur de la première colonne par une date correspondante
data.index = date_range[:len(data)]

# Afficher la série de données mise à jour
print("new data series based on test ",
      data)

##################RMSE MEAN######################
actual_values = interpolated_series['2018-12-01':].values

from sklearn.metrics import mean_squared_error

# Calculer le RMSE
rmse = mean_squared_error(actual_values, pred, squared=False)

# Afficher le RMSE
print(f"RMSE : {rmse}")


mean_error = np.mean(np.abs(actual_values - pred))
print(f"mean : {mean_error}")



####################FUTURE DATES ARIMA NON NORMALISE#########################


import pandas as pd
import matplotlib.pyplot as plt

model2 = ARIMA(interpolated_series, order=(0, 2, 0))
model2 = model2.fit()
interpolated_series.tail()

start_date = '2023-01-01'
end_date = '2040-01-01'

index_future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Fréquence mensuelle
pred = model2.predict(start=len(interpolated_series), end=len(interpolated_series) + len(index_future_dates) - 1, typ='levels').rename("ARIMA Predictions")
pred.index = index_future_dates

# Affichage des prédictions
plt.figure(figsize=(12, 5))
plt.plot(interpolated_series, label='Observed', color='blue')
plt.plot(pred, label='ARIMA Predictions', color='red')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('ARIMA Predictions')

plt.grid(True)  # Ajout du quadrillage
plt.legend()
plt.show()


####################FUTURE DATES ARIMA NORMALISE#########################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Modèle ARIMA
model2 = ARIMA(interpolated_series, order=(0, 2, 1))
model2 = model2.fit()

# Prédictions sur les données existantes
actual_values = interpolated_series['2018-12-01':].values
pred_existing = model2.predict(start=0, end=len(actual_values) - 1, typ='levels')

# Prédictions sur les futures dates
start_date = '2023-01-01'
end_date = '2040-01-01'
index_future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
pred_future = model2.predict(start=len(actual_values), end=len(actual_values) + len(index_future_dates) - 1, typ='levels')

# Normalisation des valeurs de chaque partie séparément
min_existing = np.min(actual_values)
max_existing = np.max(actual_values)
scaled_existing = (actual_values - min_existing) / (max_existing - min_existing)

min_future = np.min(pred_future)
max_future = np.max(pred_future)
scaled_future = (pred_future - min_future) / (max_future - min_future)

# Ajustement des valeurs normalisées de la partie prévisionnelle pour convergence horizontale
scaled_future_adjusted = scaled_existing[-1] - (scaled_existing[-1] - scaled_future) * (len(scaled_existing) / len(scaled_future))

# Fusion des valeurs normalisées
scaled_values = np.concatenate((scaled_existing, scaled_future_adjusted))

# Créer une liste de dates correspondant aux prévisions
pred_dates = pd.date_range(start='2018-12-01', periods=len(scaled_values), freq='MS')
from scipy.ndimage import gaussian_filter1d

# Lissage des valeurs
smoothed_values = gaussian_filter1d(scaled_values, sigma=15)


plt.figure(figsize=(12, 5))

# Trouver l'index correspondant à l'abscisse 2023-01-01
index_2023 = np.where(pred_dates == '2023-01-01')[0][0]

plt.plot(pred_dates[:index_2023], smoothed_values[:index_2023], label='Smoothed ARIMA Predictions', color='blue')
plt.plot(pred_dates[index_2023:], smoothed_values[index_2023:], color='red')

plt.xlabel('Date')
plt.ylabel('Normalized Value (0-1)')
plt.title('Smoothed ARIMA Predictions')

plt.grid(True)
plt.legend()
plt.show()




"PREVISIONS STOCKEES DANS pred_future"


############################TO CSV###################################
"""
import pandas as pd

# Créer un DataFrame à partir des données
data = pd.DataFrame({'Date': pred_dates, 'Smoothed_ARIMA_Predictions': smoothed_values})

# Exporter le DataFrame au format CSV avec le bon format de date
data.to_csv('smoothed_predictions.csv', index=False, date_format='%Y-%m-%d')
"""

