#################################################################################
####################################DATAFRAMES####################################

import json
import pandas as pd

#############CONNAITRE ESPACEMENT ENTRE CHAQUE VALEUR#########################

# Charger les données à partir du fichier JSON
with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/blocks-size.json', 'r') as f:
    data = json.load(f)


# Créer un DataFrame à partir des données
df = pd.DataFrame(data['data'])

# Convertir la colonne 'x' en format de date
df['x'] = pd.to_datetime(df['x'], unit='ms')

# Calculer la différence entre les dates consécutives
diff_days = df['x'].diff().dt.days

# Afficher les statistiques sur les différences de jours
print(diff_days.describe())


###############RECUPERER VALEUR BLOCKSIZE DANS DATAFRAME##################
#DATAFRAME=taille_blockchain



def calculate_yearly_median(json_file):
    # Charger les données à partir du fichier JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extraire les données x et y
    x_values = [pd.to_datetime(item['x'], unit='ms').year for item in data['data']]
    y_values = [item['y'] for item in data['data']]

    # Créer un DataFrame avec les données
    df = pd.DataFrame({'Year': x_values, 'Value': y_values})

    # Calculer la médiane par année
    blockchain_size = df.groupby('Year')['Value'].median().reset_index()

    return blockchain_size
    
    
blockchain_size = calculate_yearly_median('C:/Users/maxim/OneDrive/Bureau/btc/jsons/blocks-size.json')
"""
for index, row in result.iterrows():
    print("Année:", row['Year'], "Médiane:", row['Value'])
"""
print(blockchain_size)



###############RECUPERER VALEUR ELECTRICITY DANS DATAFRAME##################
#DATAFRAME=electricity_price

# Lecture du fichier JSON
with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/electricity.json', 'r') as f:
    data = json.load(f)

# Extraction des données x et y
x_values = [item['x'] for item in data['data']]
y_values = [item['y'] for item in data['data']]

# Création du DataFrame avec les données
electricity_price = pd.DataFrame({'Year': x_values, 'Price': y_values})

# Affichage du DataFrame
print(electricity_price)



###############RECUPERER VALEUR ANTIVIRUS DANS DATAFRAME##################
#DATAFRAME=antivirus_price


# Lecture du fichier JSON
with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/antivirus_price.json', 'r') as f:
    data = json.load(f)

# Extraction des données x et y
x_values = [item['x'] for item in data['data']]
y_values = [item['y'] for item in data['data']]

# Création du DataFrame avec les données
antivirus_price = pd.DataFrame({'Year': x_values, 'Price': y_values})

# Affichage du DataFrame
print(antivirus_price)



###############RECUPERER VALEUR GB_COST DANS DATAFRAME##################
#DATAFRAME=gb_cost


# Lecture du fichier JSON
with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/gbcost.json', 'r') as f:
    data = json.load(f)

# Extraction des données x et y
x_values = [item['x'] for item in data['data']]
y_values = [item['y'] for item in data['data']]

# Création du DataFrame avec les données
gb_cost = pd.DataFrame({'Year': x_values, 'Price': y_values})

# Affichage du DataFrame
print(gb_cost)



###############RECUPERER VALEUR FIREWALL DANS DATAFRAME##################
#DATAFRAME=firewall_price


# Lecture du fichier JSON
with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/price_firewall.json', 'r') as f:
    data = json.load(f)

# Extraction des données x et y
x_values = [item['x'] for item in data['data']]
y_values = [item['y'] for item in data['data']]

# Création du DataFrame avec les données
firewall_price = pd.DataFrame({'Year': x_values, 'Price': y_values})

# Affichage du DataFrame
print(firewall_price)


###############RECUPERER VALEUR INTERNET DANS DATAFRAME##################
#DATAFRAME=internet_price


# Lecture du fichier JSON
with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/internetPerGb_price.json', 'r') as f:
    data = json.load(f)

# Extraction des données x et y
x_values = [item['x'] for item in data['data']]
y_values = [item['y'] for item in data['data']]

# Création du DataFrame avec les données
internet_price = pd.DataFrame({'Year': x_values, 'Price': y_values})

# Affichage du DataFrame
print(internet_price)





#############################TRACER GRAPHES############################################


import json
import pandas as pd
import matplotlib.pyplot as plt

def generate_graphs():
    # Charger les données à partir du fichier blocks-size.json
    with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/blocks-size.json', 'r') as f:
        data = json.load(f)

    # Créer un DataFrame à partir des données
    df = pd.DataFrame(data['data'])

    # Convertir la colonne 'x' en format de date
    df['x'] = pd.to_datetime(df['x'], unit='ms')

    # Calculer la différence entre les dates consécutives
    diff_days = df['x'].diff().dt.days

    # Afficher les statistiques sur les différences de jours
    print(diff_days.describe())

    # Afficher le graphique
    plt.plot(df['x'], df['y'])
    plt.xlabel('Date')
    plt.ylabel('Block Size')
    plt.title('Block Size Over Time')
    plt.show()

    # Charger les données à partir du fichier electricity.json
    with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/electricity.json', 'r') as f:
        data = json.load(f)

    # Extraction des données x et y
    x_values = [item['x'] for item in data['data']]
    y_values = [item['y'] for item in data['data']]

    # Création du DataFrame avec les données
    electricity_price = pd.DataFrame({'Year': x_values, 'Price': y_values})

    # Afficher le graphique
    plt.plot(electricity_price['Year'], electricity_price['Price'])
    plt.xlabel('Year')
    plt.ylabel('Electricity Price')
    plt.title('Electricity Price Over Time')
    plt.show()

    # Charger les données à partir du fichier antivirus_price.json
    with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/antivirus_price.json', 'r') as f:
        data = json.load(f)

    # Extraction des données x et y
    x_values = [item['x'] for item in data['data']]
    y_values = [item['y'] for item in data['data']]

    # Création du DataFrame avec les données
    antivirus_price = pd.DataFrame({'Year': x_values, 'Price': y_values})

    # Afficher le graphique
    plt.plot(antivirus_price['Year'], antivirus_price['Price'])
    plt.xlabel('Year')
    plt.ylabel('Antivirus Price')
    plt.title('Antivirus Price Over Time')
    plt.show()

    # Charger les données à partir du fichier gbcost.json
    with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/gbcost.json', 'r') as f:
        data = json.load(f)

    # Extraction des données x et y
    x_values = [item['x'] for item in data['data']]
    y_values = [item['y'] for item in data['data']]

    # Création du DataFrame avec les données
    gb_cost = pd.DataFrame({'Year': x_values, 'Price': y_values})

    # Afficher le graphique
    plt.plot(gb_cost['Year'], gb_cost['Price'])
    plt.xlabel('Year')
    plt.ylabel('GB Cost')
    plt.title('GB Cost Over Time')
    plt.show()

    # Charger les données à partir du fichier price_firewall.json
    with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/price_firewall.json', 'r') as f:
        data = json.load(f)

    # Extraction des données x et y
    x_values = [item['x'] for item in data['data']]
    y_values = [item['y'] for item in data['data']]

                # Création du DataFrame avec les données
    firewall_price = pd.DataFrame({'Year': x_values, 'Price': y_values})

    # Afficher le graphique
    plt.plot(firewall_price['Year'], firewall_price['Price'])
    plt.xlabel('Year')
    plt.ylabel('Firewall Price')
    plt.title('Firewall Price Over Time')
    plt.show()

    # Charger les données à partir du fichier internetPerGb_price.json
    with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/internetPerGb_price.json', 'r') as f:
        data = json.load(f)

    # Extraction des données x et y
    x_values = [item['x'] for item in data['data']]
    y_values = [item['y'] for item in data['data']]

    # Création du DataFrame avec les données
    internet_price = pd.DataFrame({'Year': x_values, 'Price': y_values})

    # Afficher le graphique
    plt.plot(internet_price['Year'], internet_price['Price'])
    plt.xlabel('Year')
    plt.ylabel('Internet Price per GB')
    plt.title('Internet Price per GB Over Time')
    plt.show()

generate_graphs()


#############################TRACER GRAPHE NORMALISE#################################


from sklearn.preprocessing import MinMaxScaler

def normalize_graphs():
    # Charger les données à partir du fichier blocks-size.json
    with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/blocks-size.json', 'r') as f:
        data = json.load(f)

    # Créer un DataFrame à partir des données
    df = pd.DataFrame(data['data'])

    # Convertir la colonne 'x' en format de date
    df['x'] = pd.to_datetime(df['x'], unit='ms')

    # Afficher le graphique non normalisé
    plt.plot(df['x'], df['y'])
    plt.xlabel('Date')
    plt.ylabel('Block Size')
    plt.title('Block Size Over Time')
    plt.show()
"""
    # Charger les données à partir des fichiers JSON correspondants
    data_files = ['C:/Users/maxim/OneDrive/Bureau/btc/jsons/electricity.json', 'C:/Users/maxim/OneDrive/Bureau/btc/jsons/antivirus_price.json', 'C:/Users/maxim/OneDrive/Bureau/btc/jsons/gbcost.json', 'C:/Users/maxim/OneDrive/Bureau/btc/jsons/price_firewall.json', 'C:/Users/maxim/OneDrive/Bureau/btc/jsons/internetPerGb_price.json']

    for file in data_files:
        with open(file, 'r') as f:
            data = json.load(f)

        # Extraction des données x et y
        x_values = [item['x'] for item in data['data']]
        y_values = [item['y'] for item in data['data']]

        # Création du DataFrame avec les données
        df = pd.DataFrame({'Year': x_values, 'Price': y_values})

        # Normalisation des valeurs entre 0 et 1
        scaler = MinMaxScaler()
        df['Price'] = scaler.fit_transform(df[['Price']])

        # Afficher le graphique normalisé
        plt.plot(df['Year'], df['Price'])
        plt.xlabel('Year')
        plt.ylabel('Normalized Price')
        plt.title(file.split('.')[0].capitalize() + ' Over Time (Normalized)')
        plt.show()

normalize_graphs()
"""


# Charger les données à partir des fichiers JSON correspondants
data_files = ['C:/Users/maxim/OneDrive/Bureau/btc/jsons/electricity.json', 'C:/Users/maxim/OneDrive/Bureau/btc/jsons/antivirus_price.json', 'C:/Users/maxim/OneDrive/Bureau/btc/jsons/gbcost.json', 'C:/Users/maxim/OneDrive/Bureau/btc/jsons/price_firewall.json', 'C:/Users/maxim/OneDrive/Bureau/btc/jsons/internetPerGb_price.json']

for file in data_files:
    with open(file, 'r') as f:
        data = json.load(f)

    # Extraction des données x et y
    x_values = [item['x'] for item in data['data']]
    y_values = [item['y'] for item in data['data']]

    # Création du DataFrame avec les données
    df = pd.DataFrame({'Year': x_values, 'Price': y_values})

    # Normalisation des valeurs entre 0 et 1
    scaler = MinMaxScaler()
    df['Price'] = scaler.fit_transform(df[['Price']])

    # Obtenir le nom du fichier en extrayant le nom de base
    file_name = file.split('/')[-1].split('.')[0]

    # Afficher le graphique normalisé avec le nom du fichier comme titre
    plt.plot(df['Year'], df['Price'])
    plt.xlabel('Year')
    plt.ylabel('Normalized Price')
    plt.title(file_name.capitalize() + ' Over Time (Normalized)')
    plt.show()

############################EXPORT ALL TO CSV#########################
#print(blockchain_size)
#print(electricity_price)
#print(antivirus_price)
#print(gb_cost)
#print(firewall_price)
#print(internet_price)


#blockchain_size.to_csv('blockchain_size.csv', index=False)
#electricity_price.to_csv('electricity_price.csv', index=False)
#antivirus_price.to_csv('antivirus_price.csv', index=False)
#gb_cost.to_csv('gb_cost.csv', index=False)
#firewall_price.to_csv('firewall_price.csv', index=False)
#internet_price.to_csv('internet_price.csv', index=False)








##################################################"

"""

import csv

input_file = 'btc.csv'
output_file = 'btc_price.csv'

roi_data = []

# Lecture du fichier CSV d'entrée
with open(input_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Ignorer l'en-tête du fichier CSV
    
    # Recherche de l'index de la colonne "ROI1yr"
    roi_index = header.index('PriceUSD')
    
    for row in csv_reader:
        date = row[0]  # Supposons que la date soit dans la première colonne du fichier CSV
        roi = row[roi_index]
        roi_data.append([date, roi])

# Écriture des données extraites dans un nouveau fichier CSV
with open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Date', 'priceUSD'])  # Écriture de l'en-tête
    csv_writer.writerows(roi_data)

print("Extraction des données ROI terminée. Les données ont été enregistrées dans le fichier", output_file)
"""
