import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

np.random.seed(77)
nj = 10000


#Dado de 12 faces
dado12 = np.random.randint(1, 13, size=nj)
valores12, contagem12 = np.unique(dado12, return_counts=True)
valor_esperado12 = np.mean(dado12)


#2 Dados de 6 faces
dados6 = np.random.randint(1, 7, size=(nj, 2))
soma_dados6 = np.sum(dados6, axis=1)
valores_dados6, contagem_dados6 = np.unique(soma_dados6, return_counts=True)
valor_esperado2d6 = np.mean(soma_dados6)

plt.figure(figsize=(15, 6))

#Gráfico do dado de 12 faces
plt.subplot(1, 2, 1)
sb.barplot(x = valores12, y = contagem12, palette="Blues_d")
plt.title(f'1 dado de 12 faces\nValor esperado: {valor_esperado12:.2f}')
plt.xlabel('Face do dado')
plt.ylabel('Numero de ocorrencias')
plt.xticks(range(12), range(1, 13))

#Gráfico dos 2 dados de 6 faces
plt.subplot(1, 2, 2)
sb.barplot(x = valores_dados6, y = contagem_dados6, palette='Greens_d')
plt.title(f'2 dados de 6 faces\nValor esperado: {valor_esperado2d6:.2f}')
plt.xlabel('Soma das faces')
plt.ylabel('Número de ocorrências')
plt.xticks(range(11), range(2, 13))

plt.tight_layout()
plt.show()
