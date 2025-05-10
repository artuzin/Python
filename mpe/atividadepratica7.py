import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Carregando o dataset diamonds.csv
dados = pd.read_csv('C:\\Usuários\\artur\\OneDrive\\Documentos\\GitHub\\Python\\mpe\\diamonds.csv')

# Exercício 1: Divida a população, selecionando os casos pela variável 'cut', em amostras aleatórias de 20%
print("=== Exercício 1 ===")
# Agrupando por 'corte' e amostrando 20% de cada grupo
amostra_20 = dados.groupby('cut').apply(lambda x: x.sample(frac=0.2, random_state=42)).reset_index(drop=True)
print(f"Tamanho da amostra (20%): {len(amostra_20)}")
print(amostra_20.head())

# Exercício 2: Faça uma amostra com 2000 entradas, mantendo a proporção da variável 'cut' tal qual o arquivo original
print("\n=== Exercício 2 ===")
# Calculando as proporções de cada categoria em 'cut'
proporcoes_corte = dados['cut'].value_counts(normalize=True)
# Determinando o número de entradas por categoria para 2000 amostras
tamanho_amostra = 2000
contagem_corte = (proporcoes_corte * tamanho_amostra).round().astype(int)
# Ajustando para garantir que o total seja exatamente 2000
soma_contagem = contagem_corte.sum()
if soma_contagem < tamanho_amostra:
    contagem_corte.iloc[0] += tamanho_amostra - soma_contagem  # Adiciona a diferença à primeira categoria
elif soma_contagem > tamanho_amostra:
    contagem_corte.iloc[0] -= soma_contagem - tamanho_amostra  # Remove a diferença da primeira categoria
# Amostrando com base nas proporções
amostra_2000 = pd.concat([
    dados[dados['cut'] == corte].sample(n=contagem, replace=False, random_state=42)
    for corte, contagem in contagem_corte.items()
]).sample(n=tamanho_amostra, random_state=42)  # Garante exatamente 2000 entradas
print(f"Tamanho da amostra: {len(amostra_2000)}")
print("Proporções no arquivo original:")
print(proporcoes_corte)
print("Proporções na amostra de 2000:")
print(amostra_2000['cut'].value_counts(normalize=True))

# Exercício 3: Plote histogramas para a variável 'price' do arquivo original e de todas as amostras
print("\n=== Exercício 3 ===")
# Plotando histogramas
plt.figure(figsize=(15, 10))

# Histograma do dataset original
plt.subplot(3, 1, 1)
plt.hist(dados['price'], bins=50, color='blue', alpha=0.7)
plt.title('Histograma de Preço - Dataset Original')
plt.xlabel('Preço')
plt.ylabel('Frequência')

# Histograma da amostra de 20%
plt.subplot(3, 1, 2)
plt.hist(amostra_20['price'], bins=50, color='green', alpha=0.7)
plt.title('Histograma de Preço - Amostra 20%')
plt.xlabel('Preço')
plt.ylabel('Frequência')

# Histograma da amostra de 2000 entradas
plt.subplot(3, 1, 3)
plt.hist(amostra_2000['price'], bins=50, color='red', alpha=0.7)
plt.title('Histograma de Preço - Amostra 2000 Entradas')
plt.xlabel('Preço')
plt.ylabel('Frequência')

plt.tight_layout()
plt.savefig('histogramas_preco.png')

# Comentário sobre os resultados
print("Os histogramas mostram que as distribuições de 'preço' nas amostras são semelhantes à do dataset original, com tendência de assimetria à direita. A amostra de 20% reflete melhor a variabilidade original, enquanto a amostra de 2000 pode ter menos variabilidade devido ao tamanho fixo.")

# Problemas Motivadores Intervalos de Confiança

# Problema 1: Intervalo de confiança para a vida média de válvulas
print("\n=== Problema 1 ===")
# Dados fornecidos
tamanho_amostra_1 = 400  # tamanho da amostra
media_amostra_1 = 800  # vida média da amostra
desvio_padrao_1 = 100  # desvio padrão da amostra

# (a) Intervalo de confiança de 99% para a vida média da população
z_99 = stats.norm.ppf(0.995)  # z-score para 99% de confiança (bicaudal)
margem_erro_99 = z_99 * (desvio_padrao_1 / np.sqrt(tamanho_amostra_1))
limite_inferior_99 = media_amostra_1 - margem_erro_99
limite_superior_99 = media_amostra_1 + margem_erro_99
print(f"(a) Intervalo de confiança de 99%: ({limite_inferior_99:.2f}, {limite_superior_99:.2f})")

# (b) Com que confiança pode-se dizer que a vida média é 800 ± 0.98?
margem_erro_dada = 0.98
z_dado = margem_erro_dada / (desvio_padrao_1 / np.sqrt(tamanho_amostra_1))
nivel_confianca = 2 * stats.norm.cdf(z_dado) - 1  # Confiança bicaudal
print(f"(b) Confiança para 800 ± 0.98: {max(0, min(1, nivel_confianca)) * 100:.2f}%")

# (c) Tamanho da amostra para intervalo de confiança de 95% com 800 ± 7.84
z_95 = stats.norm.ppf(0.975)  # z-score para 95% de confiança
margem_erro_alvo = 7.84
tamanho_necessario = (z_95 * desvio_padrao_1 / margem_erro_alvo) ** 2
print(f"(c) Tamanho da amostra necessário: {int(np.ceil(tamanho_necessario))}")

# Problema 2: Intervalo de confiança para a média dos diâmetros de rolamentos
print("\n=== Problema 2 ===")
# Dados fornecidos
tamanho_amostra_2 = 200
media_amostra_2 = 0.824
desvio_padrao_2 = 0.042

# (a) Intervalo de confiança de 95%
z_95 = stats.norm.ppf(0.975)
margem_erro_95 = z_95 * (desvio_padrao_2 / np.sqrt(tamanho_amostra_2))
limite_inferior_95 = media_amostra_2 - margem_erro_95
limite_superior_95 = media_amostra_2 + margem_erro_95
print(f"(a) Intervalo de confiança de 95%: ({limite_inferior_95:.4f}, {limite_superior_95:.4f})")

# (b) Intervalo de confiança de 99%
z_99 = stats.norm.ppf(0.995)
margem_erro_99 = z_99 * (desvio_padrao_2 / np.sqrt(tamanho_amostra_2))
limite_inferior_99 = media_amostra_2 - margem_erro_99
limite_superior_99 = media_amostra_2 + margem_erro_99
print(f"(b) Intervalo de confiança de 99%: ({limite_inferior_99:.4f}, {limite_superior_99:.4f})")

# Problema 3: Probabilidade binomial para o lançamento de uma moeda
print("\n=== Problema 3 ===")
# Dados fornecidos
tamanho_amostra_3 = 50
quantidade_caras = 36
probabilidade_cara = 0.5  # Probabilidade de cara (moeda honesta)

# Probabilidade de obter exatamente 36 caras
probabilidade_exata = stats.binom.pmf(quantidade_caras, tamanho_amostra_3, probabilidade_cara)
print(f"Probabilidade de exatamente 36 caras: {probabilidade_exata:.6f}")

# Para verificar se a moeda é honesta, calculamos a probabilidade de 36 ou mais caras (teste unilateral)
prob_36_ou_mais = 1 - stats.binom.cdf(quantidade_caras - 1, tamanho_amostra_3, probabilidade_cara)
print(f"Probabilidade de 36 ou mais caras: {prob_36_ou_mais:.6f}")
print("Como a probabilidade é muito baixa (menor que 0.05), sugere-se rejeitar a hipótese de que a moeda é honesta.")

# Problema 4: Teste de hipóteses para médias de duas turmas
print("\n=== Problema 4 ===")
# Dados fornecidos
tamanho_amostra_4a, tamanho_amostra_4b = 40, 50
media_amostra_4a, media_amostra_4b = 7.4, 7.8
desvio_padrao_4a, desvio_padrao_4b = 0.8, 0.7

# Teste de hipóteses (diferença entre médias)
# Hipótese nula: mu_a - mu_b = 0
# Hipótese alternativa: mu_a != mu_b
diferenca_medias = media_amostra_4a - media_amostra_4b
erro_padrao = np.sqrt((desvio_padrao_4a**2 / tamanho_amostra_4a) + (desvio_padrao_4b**2 / tamanho_amostra_4b))
estatistica_z = diferenca_medias / erro_padrao
valor_p = 2 * (1 - stats.norm.cdf(abs(estatistica_z)))  # p-valor bicaudal

# (a) Nível de significância 0.05 (95% de confiança)
print(f"(a) p-valor para significância 0.05: {valor_p:.4f}")
print("Como o p-valor é menor que 0.05, rejeita-se a hipótese nula: há diferença significativa entre as médias.")

# (b) Nível de significância 0.01 (99% de confiança)
print(f"(b) p-valor para significância 0.01: {valor_p:.4f}")
print("Como o p-valor é maior que 0.01, não se rejeita a hipótese nula no nível de 99% de confiança.")