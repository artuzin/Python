import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm

# Parâmetros
lambdas = [1, 4, 10, 30]  # Diferentes valores de lambda para testar
num_samples = 10000  # Número de amostras

# Configuração dos subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Distribuição de Poisson vs Ajuste Normal para diferentes λ', fontsize=16)

for i, lambda_ in enumerate(lambdas):
    # Gerar amostras da distribuição de Poisson
    poisson_data = np.random.poisson(lambda_, num_samples)
    
    # Calcular histograma
    counts, bins, _ = axes[i//2, i%2].hist(poisson_data, bins=range(min(poisson_data), max(poisson_data) + 2), 
                                           density=True, alpha=0.7, color='skyblue', label='Poisson')
    
    # Ajustar curva normal
    mu, sigma = lambda_, np.sqrt(lambda_)  # Para Poisson, média = var = λ
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    normal_fit = norm.pdf(x, mu, sigma)
    axes[i//2, i%2].plot(x, normal_fit, 'r-', label='Normal Fit')
    
    # Plotar PMF teórica de Poisson
    x_poisson = np.arange(0, max(poisson_data) + 1)
    poisson_pmf = poisson.pmf(x_poisson, lambda_)
    axes[i//2, i%2].plot(x_poisson, poisson_pmf, 'bo', ms=4, label='Poisson PMF')
    axes[i//2, i%2].vlines(x_poisson, 0, poisson_pmf, colors='b', lw=2, alpha=0.5)
    
    axes[i//2, i%2].set_title(f'λ = {lambda_}')
    axes[i//2, i%2].legend()
    axes[i//2, i%2].grid(True)

plt.tight_layout()
plt.show()

# Parâmetros para comparação das distribuições
n = 100  # Número de tentativas para binomial
p = 0.1  # Probabilidade de sucesso para binomial
lambda_poisson = n * p  # λ para Poisson (deve ser igual a n*p para comparação)
mu_normal = n * p  # Média para normal
sigma_normal = np.sqrt(n * p * (1 - p))  # Desvio padrão para normal

# Gerar dados
binomial_data = np.random.binomial(n, p, 10000)
poisson_data = np.random.poisson(lambda_poisson, 10000)
normal_data = np.random.normal(mu_normal, sigma_normal, 10000)

# Plotar histogramas
plt.figure(figsize=(10, 6))
plt.hist(binomial_data, bins=30, density=True, alpha=0.6, color='blue', label='Binomial')
plt.hist(poisson_data, bins=30, density=True, alpha=0.6, color='green', label='Poisson')
plt.hist(normal_data, bins=30, density=True, alpha=0.6, color='red', label='Normal')

# Adicionar legendas e título
plt.title('Comparação entre Distribuições Binomial, Poisson e Normal\n(n=100, p=0.1, λ=10)')
plt.xlabel('Valor')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.show()