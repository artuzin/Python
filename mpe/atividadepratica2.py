import pandas as pd
import matplotlib as plt
import seaborn as sb

dados = pd.read_csv('C:\\Usuários\\artur\\OneDrive\\Documentos\\GitHub\\Python\\mpe\\TabelaSalarial.csv')

dados = dados.drop(columns=["N"])

colunas_numeros = dados.select_dtypes(include=['number'])

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

