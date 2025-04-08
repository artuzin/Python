import numpy as np
import scipy.stats as stats
import seaborn as sb
import matplotlib.pyplot as plt

media = 3
dp = 1

prob_menos_2kg = stats.norm.cdf(2, loc=media, scale=dp)

n_partos = 20
p = prob_menos_2kg

bebes_esperados = n_partos * p

print(f"Probabilidade de um bebê pesar menos de 2kg: {prob_menos_2kg:.4f}")
print(f"Número esperado de bebês abaixo de 2kg por dia: {bebes_esperados:.2f}")

valores_x = np.arange(0, n_partos + 1)

probabilidades = stats.binom.pmf(valores_x, n_partos, p)

plt.figure(figsize=(10,6))
sb.barplot(x=valores_x, y=probabilidades, color='blue')

plt.xlabel("Número de bebês abaixo de 2kg por dia")
plt.ylabel("Probabilidades")
plt.title("Distirbuição do número de bebês abaixo de 2kg por dia")
plt.xticks(valores_x)

plt.show()