import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Daten importieren
data = pd.read_csv("candy-data.csv")

#Überblick verschaffen
print(data.head())
print(data.describe())

# nach Winpercent sortieren und die fünf besten Süßigkeiten anzeigen lassen
sorted_data = data.sort_values(by="winpercent", ascending=False)
only_name_win_data = sorted_data[['competitorname', 'winpercent']]
print(only_name_win_data.head())

# Nur numerische Spalten auswählen
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Korrelation der Merkmale nach Prozentsatz der Gewinne im Head-to-Head-Vergleich absteigend berechnen
corr_data = numeric_data.corr()
correlations_win = corr_data['winpercent'].sort_values(ascending=False)
print(correlations_win)


# Korrelation zwischen den Variablen in eienr Matrix visualisieren
plt.figure(figsize=(14, 10))
sns.heatmap(data=corr_data, annot=True, cmap="coolwarm")
plt.show()

# Regression durchführen
X = numeric_data.drop("winpercent", axis=1) # nur unabhängige Variablen in X
X = sm.add_constant(X)  # Konstante (intercept) hinzufügen
y = numeric_data["winpercent"] # abhängige Variable bestimmen

model = sm.OLS(y, X).fit() # Regression mithilfe der Methode der kleinsten Quadrate (OLS)
print(model.summary())

