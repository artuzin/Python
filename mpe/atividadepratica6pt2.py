
# Importação das bibliotecas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom

# Configurações iniciais
sns.set(style="whitegrid")

# PARTE 1: Distribuição binomial do tipo sanguíneo A (n=10, p=0.4)
n = 10
p = 0.4
x_vals = np.arange(0, n+1)
probabilidades = binom.pmf(x_vals, n, p)

df_sangue = pd.DataFrame({
    'Número de pessoas com tipo A': x_vals,
    'Probabilidade': probabilidades
})

# PARTE 2: P(X = 2)
prob_2 = binom.pmf(2, n, p)
print(f"2) P(X = 2) = {prob_2:.5f}")

# PARTE 3: Gráfico de barras da distribuição
plt.figure(figsize=(10, 6))
sns.barplot(data=df_sangue, x='Número de pessoas com tipo A', y='Probabilidade', palette='Blues_d')
plt.title('Distribuição Binomial: Tipo Sanguíneo A (n=10, p=0.4)')
plt.xlabel('Número de Pessoas com Tipo A')
plt.ylabel('Probabilidade')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# PARTE 4: P(X <= 3)
prob_ate_3 = binom.cdf(3, n, p)
print(f"4) P(X ≤ 3) = {prob_ate_3:.5f}")

# PARTE 5: Probabilidade de sair exatamente 3 caras em 5 lançamentos
prob_cara_3x = binom.pmf(3, 5, 0.5)
print(f"5) P(3 caras em 5 lançamentos) = {prob_cara_3x:.5f}")

# PARTE 6: A probabilidade é razoável (distribuição simétrica, valores centrais têm maior chance)

# PARTE 7: Gráfico da distribuição
x_moeda_nv = np.arange(0, 6)
y_moeda_nv = binom.pmf(x_moeda_nv, 5, 0.5)
df_moeda_nv = pd.DataFrame({
    'Número de caras': x_moeda_nv,
    'Probabilidade': y_moeda_nv
})

plt.figure(figsize=(8, 5))
sns.barplot(data=df_moeda_nv, x='Número de caras', y='Probabilidade', palette='Greens')
plt.title('Distribuição: Moeda Não-Viciada (n=5, p=0.5)')
plt.xlabel('Número de Caras')
plt.ylabel('Probabilidade')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# PARTE 8: Distribuição com moeda viciada
x_moeda_viciada = np.arange(0, 7)
y_moeda_viciada = binom.pmf(x_moeda_viciada, 6, 0.6)
df_moeda_viciada = pd.DataFrame({
    'Número de caras': x_moeda_viciada,
    'Probabilidade': y_moeda_viciada
})

# PARTE 9: Gráfico da moeda viciada
plt.figure(figsize=(8, 5))
sns.barplot(data=df_moeda_viciada, x='Número de caras', y='Probabilidade', palette='Reds')
plt.title('Distribuição: Moeda Viciada (n=6, p=0.6)')
plt.xlabel('Número de Caras')
plt.ylabel('Probabilidade')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# PARTE 10: Média e variância
media_mv = binom.mean(6, 0.6)
variancia_mv = binom.var(6, 0.6)
print(f"10) Média = {media_mv:.2f}")
print(f"    Variância = {variancia_mv:.2f}")

