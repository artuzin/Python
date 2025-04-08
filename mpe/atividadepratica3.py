import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

dados = pd.read_csv('TabelaSalarial.csv')

dados = dados.drop(columns=["N"])

colunas_numeros = dados.select_dtypes(include=['number'])

sb.set(style='whitegrid')

print(dados.head())

mean_values = colunas_numeros.mean()
median_values = colunas_numeros.median()
std_values = colunas_numeros.std()
var_values = colunas_numeros.var()
range_values = colunas_numeros.max() - colunas_numeros.min()

print("Médias: \n", mean_values)
print("Medianas: \n", median_values)
print("Desvio Padrão: \n", std_values)
print("Variâncias: \n", var_values)
print("Amplitude: \n", range_values)

plt.figure(figsize=(15, 10))
for i, coluna in enumerate(colunas_numeros.columns, 1):
    plt.subplot(2, 2, i)
    sb.histplot(data=colunas_numeros, x=coluna, kde=True, bins=30, color='purple')
    plt.title(f'Histograma de {coluna}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,10))
for i, coluna in enumerate(colunas_numeros.columns, 1):
    plt.subplot(2, 2, i)
    sb.boxplot(data=colunas_numeros, y=coluna, color='yellow')
    plt.title(f'Boxplot de {coluna}')   
plt.tight_layout()
plt.show()


