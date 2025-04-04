import numpy as np
import scipy.stats as stats

media = 6.7
dp = 1.2

prob_6_pts = stats.norm.pdf(6, loc=media, scale=dp)

percentil10 = stats.norm.ppf (0.1, loc=media, scale=dp)

percentil90 = stats.norm.ppf(0.9, loc=media, scale=dp)

print(f"Probabilidade de um aluno tirar exatamente 6 pontos: {prob_6_pts:.4f}")
print(f"Grau máximo dos 10% mais baixo da sala: {percentil10:.2f}")
print(f"Grau mínimo dos 10% mais altos da classe: {percentil90:.2f}")