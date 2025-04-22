import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy.stats import binom

#Questão 1
#Letra A
# Dados do problema
n1 = 2000
p1 = 0.001
lambda1_ = n1 * p1  # Parâmetro de Poisson

# Probabilidade de exatamente 3 sofrerem reação
prob_3 = poisson.pmf(3, lambda1_)
print(f"(a) Probabilidade de exatamente 3: {prob_3:.4f} ou {prob_3*100:.2f}%")

#Letra B
# Probabilidade de mais que 2 sofrerem reação (1 - P(0) - P(1) - P(2))
prob_mais_que_2 = 1 - poisson.cdf(2, lambda1_)
print(f"(b) Probabilidade de mais que 2: {prob_mais_que_2:.4f} ou {prob_mais_que_2*100:.2f}%")

# Análise de segurança
print("\nAnálise de segurança:")
print(f"Probabilidade de 0 reações: {poisson.pmf(0, lambda1_)*100:.2f}%")
print(f"Probabilidade de até 1 reação: {poisson.cdf(1, lambda1_)*100:.2f}%")
print("O remédio pode ser considerado seguro, pois a probabilidade de mais de 2 reações é baixa.")

#Questão 2
#Letra A
# Dados do problema
media_por_100k = 3
populacao = 200000
lambda2_ = media_por_100k * (populacao / 100000)

# Probabilidade de 0 acidentes
prob1_0 = poisson.pmf(0, lambda2_)
print(f"(a) P(0 acidentes) = {prob1_0:.4f} ou {prob1_0*100:.2f}%")

#Letra B
# Probabilidade de 2 acidentes
prob_2 = poisson.pmf(2, lambda2_)
print(f"(b) P(2 acidentes) = {prob_2:.4f} ou {prob_2*100:.2f}%")

#Letra C
# Probabilidade de 6 acidentes
prob_6 = poisson.pmf(6, lambda2_)
print(f"(c) P(6 acidentes) = {prob_6:.4f} ou {prob_6*100:.2f}%")

#Letra D
# Probabilidade de 8 acidentes
prob_8 = poisson.pmf(8, lambda2_)
print(f"(d) P(8 acidentes) = {prob_8:.4f} ou {prob_8*100:.2f}%")

#Letra E
# Probabilidade de 10 acidentes
prob_10 = poisson.pmf(10, lambda2_)
print(f"(e) P(10 acidentes) = {prob_10:.4f} ou {prob_10*100:.2f}%")

#Letra F
# Probabilidade entre 4 e 10 acidentes
prob_4_a_10 = poisson.cdf(10, lambda2_) - poisson.cdf(3, lambda2_)
print(f"(f) P(4 a 10 acidentes) = {prob_4_a_10:.4f} ou {prob_4_a_10*100:.2f}%")

# Plotando o histograma
plt.figure(figsize=(10, 6))
x1 = np.arange(0, 15)
y1 = poisson.pmf(x1, lambda2_)
plt.bar(x1, y1, color='skyblue')
plt.title(f'Distribuição de Poisson para λ = {lambda2_}')
plt.xlabel('Número de acidentes')
plt.ylabel('Probabilidade')
plt.axvline(lambda2_, color='red', linestyle='--', label=f'Média (λ = {lambda2_})')
plt.legend()
plt.show()

# Análise de segurança
print("\nAnálise de segurança:")
print(f"Probabilidade de mais de 6 acidentes: {(1 - poisson.cdf(6, lambda2_))*100:.2f}%")
print("Considerando a probabilidade de mais de 6 acidentes, pode ser recomendada uma intervenção.")

#Questão 3
#Letra A
n2 = 100
p2 = 0.03
lambda3_ = n2 * p2

prob2_0 = poisson.pmf(0, lambda3_)
print(f"(a) P(0 defeituosas) = {prob2_0:.4f} ou {prob2_0*100:.2f}%")

#Letra B
prob_1 = poisson.pmf(1, lambda3_)
print(f"(b) P(1 defeituosa) = {prob_1:.4f} ou {prob_1*100:.2f}%")

#Letra C
prob_2 = poisson.pmf(2, lambda3_)
print(f"(c) P(2 defeituosas) = {prob_2:.4f} ou {prob_2*100:.2f}%")

#Letra D
prob_3 = poisson.pmf(3, lambda3_)
print(f"(d) P(3 defeituosas) = {prob_3:.4f} ou {prob_3*100:.2f}%")

#Letra E
prob_4 = poisson.pmf(4, lambda3_)
print(f"(e) P(4 defeituosas) = {prob_4:.4f} ou {prob_4*100:.2f}%")

# Plotando o histograma
plt.figure(figsize=(10, 6))
x2 = np.arange(0, 10)
y2 = poisson.pmf(x2, lambda3_)
plt.bar(x2, y2, color='lightgreen')
plt.title(f'Distribuição de Poisson para λ = {lambda3_} (3% de defeitos em 100 peças)')
plt.xlabel('Número de peças defeituosas')
plt.ylabel('Probabilidade')
plt.axvline(lambda3_, color='red', linestyle='--', label=f'Média (λ = {lambda3_})')
plt.legend()
plt.show()

# Análise de qualidade
print("\nAnálise de qualidade:")
print(f"Probabilidade de até 3 defeitos: {poisson.cdf(3, lambda3_)*100:.2f}%")
print(f"Probabilidade de mais de 4 defeitos: {(1 - poisson.cdf(4, lambda3_))*100:.2f}%")
print("Os resultados podem ser considerados satisfatórios.")

#Questão 4
#Letra A
n3 = 10
p3 = 0.1

prob_binomial = binom.pmf(2, n3, p3)
print(f"(a) Probabilidade binomial (exatamente 2 defeituosas): {prob_binomial:.4f}")

#Letra B
lambda4_ = n3 * p3
prob_poisson = poisson.pmf(2, lambda4_)
print(f"(b) Probabilidade Poisson (exatamente 2 defeituosas): {prob_poisson:.4f}")
print(f"Diferença absoluta: {abs(prob_binomial - prob_poisson):.4f}")

# Comparação teórica
print("\nCritérios para uso de Poisson em vez de Binomial:")
print("1. Quando n é grande (tipicamente n > 20)")
print("2. Quando p é pequeno (tipicamente p < 0.1)")
print("3. Quando o produto n*p é moderado (tipicamente 0.1 < n*p < 10)")

print("\nDiferença para a distribuição normal:")
print("1. Poisson e Binomial são discretas, Normal é contínua")
print("2. Poisson é assimétrica (para λ pequeno), Normal é simétrica")
print("3. Poisson modela contagens de eventos raros, Binomial modela sucessos em tentativas")