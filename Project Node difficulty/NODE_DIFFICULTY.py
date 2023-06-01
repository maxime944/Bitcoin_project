
import json
import pandas as pd
import matplotlib.pyplot as plt

def process_dataframes():
    def read_json_file(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        x_values = [item['x'] for item in data['data']]
        y_values = [item['y'] for item in data['data']]
        return pd.DataFrame({'Year': x_values, 'Value': y_values})

    # Charger les données de blocks-size.json dans un DataFrame
    with open('C:/Users/maxim/OneDrive/Bureau/btc/jsons/blocks-size.json', 'r') as f:
        data = json.load(f)
    df_blocks_size = pd.DataFrame(data['data'])
    df_blocks_size['x'] = pd.to_datetime(df_blocks_size['x'], unit='ms')
    #diff_days = df_blocks_size['x'].diff().dt.days
    #print(diff_days.describe())

    # Calculer la médiane par année pour blockchain_size
    df_blockchain_size = calculate_yearly_median('C:/Users/maxim/OneDrive/Bureau/btc/jsons/blocks-size.json')

    # Charger les données des autres fichiers JSON dans des DataFrames
    df_electricity_price = read_json_file('C:/Users/maxim/OneDrive/Bureau/btc/jsons/electricity.json')
    df_antivirus_price = read_json_file('C:/Users/maxim/OneDrive/Bureau/btc/jsons/antivirus_price.json')
    df_gb_cost = read_json_file('C:/Users/maxim/OneDrive/Bureau/btc/jsons/gbcost.json')
    df_firewall_price = read_json_file('C:/Users/maxim/OneDrive/Bureau/btc/jsons/price_firewall.json')
    df_internet_price = read_json_file('C:/Users/maxim/OneDrive/Bureau/btc/jsons/internetPerGb_price.json')

    return df_blocks_size, df_blockchain_size, df_electricity_price, df_antivirus_price, df_gb_cost, df_firewall_price, df_internet_price


def calculate_yearly_median(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    x_values = [pd.to_datetime(item['x'], unit='ms').year for item in data['data']]
    y_values = [item['y'] for item in data['data']]
    df = pd.DataFrame({'Year': x_values, 'Value': y_values})
    blockchain_size = df.groupby('Year')['Value'].median().reset_index()
    return blockchain_size


# Appel de la fonction pour obtenir les DataFrames
df_blocks_size, df_blockchain_size, df_electricity_price, df_antivirus_price, df_gb_cost, df_firewall_price, df_internet_price = process_dataframes()

# Affichage des DataFrames

print("\ndf_blockchain_size:")
print(df_blockchain_size)
print("\ndf_electricity_price:")
print(df_electricity_price)
print("\ndf_antivirus_price:")
print(df_antivirus_price)
print("\ndf_gb_cost:")
print(df_gb_cost)
print("\ndf_firewall_price:")
print(df_firewall_price)
print("\ndf_internet_price:")
print(df_internet_price)

# Renommer les colonnes des DataFrames avant la fusion
df_blockchain_size = df_blockchain_size.rename(columns={'Value': 'Blockchain Size'})
df_electricity_price = df_electricity_price.rename(columns={'Value': 'Electricity'})
df_antivirus_price = df_antivirus_price.rename(columns={'Value': 'Antivirus'})
df_gb_cost = df_gb_cost.rename(columns={'Value': 'Gb Cost'})
df_firewall_price = df_firewall_price.rename(columns={'Value': 'Firewall Price'})
df_internet_price = df_internet_price.rename(columns={'Value': 'Internet Price'})

# Fusion des DataFrames en utilisant la colonne 'x' comme clé de fusion
df_merged = pd.merge(df_blockchain_size, df_electricity_price, on='Year', how='outer')
df_merged = pd.merge(df_merged, df_antivirus_price, on='Year', how='outer')
df_merged = pd.merge(df_merged, df_gb_cost, on='Year', how='outer')
df_merged = pd.merge(df_merged, df_firewall_price, on='Year', how='outer')
df_merged = pd.merge(df_merged, df_internet_price, on='Year', how='outer')


###############################################################################"

# Fonction pour ajuster les valeurs en fonction de la baisse de 7% chaque année
def adjust_values(df):
    for column in df.columns:
        if column != 'Year':
            for i in range(1, len(df)):
                df.loc[i, column] = df.loc[i-1, column] * 0.93  # 0.93 représente une baisse de 7%

# Appel de la fonction pour ajuster les valeurs du DataFrame fusionné
adjust_values(df_merged)

# Affichage du DataFrame ajusté
print(df_merged)




def plot_normalized_metrics(df):
    # Normaliser les colonnes du DataFrame
    df_normalized = (df - df.min()) / (df.max() - df.min())

    # Tracer les métriques normalisées une par une
    plt.figure(figsize=(12, 6))
    for column in df_normalized.columns:
        # Exclure la colonne 'Year' lors du tracé
        if column != 'Year':
            plt.plot(df_normalized.index, df_normalized[column], label=column)

    # Configurer les axes et les légendes
    plt.xlabel('Années')
    plt.ylabel('Valeurs normalisées')
    plt.title('Graphique des valeurs normalisées')
    plt.xticks(df_normalized.index, df['Year'], rotation=45)

    # Ajouter le quadrillage
    plt.grid(True)

    # Déplacer les légendes en dehors du graphique
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Afficher le graphique
    plt.show()

# Appel de la fonction avec chaque DataFrame séparément
plot_normalized_metrics(df_blockchain_size)
plot_normalized_metrics(df_electricity_price)
plot_normalized_metrics(df_antivirus_price)
plot_normalized_metrics(df_gb_cost)
plot_normalized_metrics(df_firewall_price)
plot_normalized_metrics(df_internet_price)




# Affichage du DataFrame agrégé
#print(df_merged)

###########################GRAPHIQUE##############################


def trace_normalized_graph(df):
    # Normaliser les colonnes du DataFrame
    df_normalized = (df - df.min()) / (df.max() - df.min())

    # Tracer le graphique
    plt.figure(figsize=(12, 6))
    for column in df_normalized.columns:
        plt.plot(df_normalized.index, df_normalized[column], label=column)

    # Configurer les axes et les légendes
    plt.xlabel('Années')
    plt.ylabel('Valeurs normalisées')
    plt.title('Graphique des valeurs normalisées')
    plt.xticks(df_normalized.index, df['Year'], rotation=45)

    # Ajouter le quadrillage
    plt.grid(True)

    # Déplacer les légendes en dehors du graphique
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Afficher le graphique
    plt.show()

# Appel de la fonction avec le DataFrame df_merged
trace_normalized_graph(df_merged)

#################################GRAPHIQUE DIFFICULTE CREATION NOEUD##########################################

# Définir les coefficients d'importance pour chaque DataFrame (valeurs arbitraires ici)
coefficients = {
    'Blockchain Size': 0.1,
    'Electricity': 0.3,
    'Antivirus': 0.2,
    'Gb Cost': 0.1,
    'Firewall Price': 0.2,
    'Internet Price': 0.1
}

# Calculer la combinaison pondérée des valeurs
combined_curve = sum(df_merged[column] * coefficients[column] for column in df_merged.columns[1:])

# Normaliser la courbe combinée
combined_curve_normalized = (combined_curve - combined_curve.min()) / (combined_curve.max() - combined_curve.min())

# Tracer la courbe normalisée
plt.figure(figsize=(12, 6))
plt.plot(df_merged['Year'], combined_curve_normalized)
plt.xlabel('Années')
plt.ylabel('Valeurs combinées (normalisées)')
plt.title("Evolution de la difficulté de création d'un noeud BTC")
plt.grid(True)
plt.show()




import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Suppose combined_curve_normalized est votre série chronologique initiale
# Convertir la série chronologique en DataFrame
df = pd.DataFrame({'values': combined_curve_normalized})

# Créer un index numérique à partir de 0 jusqu'à la longueur de la série chronologique
df['index'] = np.arange(len(df))

# Identifier les valeurs manquantes dans la série chronologique
missing_values = df['values'].isnull()

# Créer une copie de la série chronologique pour y ajouter des valeurs intermédiaires
interpolated_values = df['values'].copy()

# Utiliser une régression linéaire pour générer des valeurs intermédiaires
regression_model = LinearRegression()

# Boucle sur les valeurs manquantes et effectue une régression linéaire pour générer des valeurs intermédiaires
for i in range(len(df)):
    if missing_values[i]:
        # Séparer les données disponibles pour la régression
        x_train = df.loc[~missing_values, 'index'].values.reshape(-1, 1)
        y_train = df.loc[~missing_values, 'values'].values

        # Ajuster le modèle de régression linéaire
        regression_model.fit(x_train, y_train)

        # Générer une valeur intermédiaire en utilisant le modèle de régression
        interpolated_value = regression_model.predict(np.array([[i]]))

        # Ajouter la valeur intermédiaire à la série chronologique interpolée
        interpolated_values[i] = interpolated_value

# Remplacer les valeurs interpolées dans la série chronologique initiale
df['values'] = interpolated_values

# Utiliser la série chronologique interpolée pour continuer l'analyse





