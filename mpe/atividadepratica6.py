import pandas as pd
import seaborn as sb
from scipy.stats import binom
import matplotlib.pyplot as plt

# 1 - Moeda Viciada
n_moedas = 1000
p_cara = 0.6
p_coroa = 1 - p_cara
coroas_esperadas = n_moedas * p_coroa
print("Problema 1: Moeda Viciada")
print(f"Número de coroas esperadas: {coroas_esperadas}")

# 2 - Quebra de ovos
n_ovos = 20
limite_quebra = 2
p_quebra = 0.05
prob_2_quebrados = binom.cdf(limite_quebra, n_ovos, p_quebra)
print("Problema 2: Quebra de Ovos")
print(f"Probabilidade de no máximo 2 ovos quebrarem: {prob_2_quebrados}")

#Gráfico da 2
x = list(range(n_ovos + 1))
y = binom.pmf(x, n_ovos, p_quebra)

df = pd.DataFrame({
    'Quebras': x,
    'Probabilidade': y
})

plt.figure(figsize=(10, 6))
sb.barplot(data=df, x='Quebras', y='Probabilidade', color='purple')
plt.title('Distribuição Binomial - Quebra de Ovos por Embalagem(n=20, p=0.05)')
plt.xlabel('Número de ovos quebrados')
plt.ylabel('Probabilidade')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()