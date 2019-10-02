import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score
import numpy as np 

def split(x,y,test_size):
    treino_x = x[:1212] 
    treino_y = y[:1212] 
    teste_x = x[1212:]
    teste_y = y[1212:]

    return treino_x, teste_x, treino_y, teste_y



uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"



dados = pd.read_csv(uri)

a_renomear = {
    "expected_hours" : 'horas_esperadas', 
    "price" : "preco", 
    'unfinished': 'nao_finalizado'
}

dados = dados.rename(columns=a_renomear)

troca = {
    0 : 1, 
    1 : 0
}


dados['finalizado'] = dados.nao_finalizado.map(troca)

#print(dados.tail())

#sns.scatterplot(x="horas_esperadas",y="preco",data=dados)
#sns.scatterplot(x="horas_esperadas",y="preco",hue="finalizado",data=dados)
#sns.relplot(x="horas_esperadas",y="preco",hue="finalizado", col="finalizado",data=dados)

#plt.show() 

x = dados[['horas_esperadas','preco']]
y = dados['finalizado']

SEED = 200 

treino_x, teste_x, treino_y, teste_y = train_test_split(x,y,test_size=0.25, stratify=y) 
print("Treinamento com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC(max_iter=2000) 
modelo.fit(treino_x,treino_y) 
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y,previsoes) * 100 
print("A acuracia foi %.2f%%" % acuracia) 

previsoes_de_base = np.ones(540)
acuracia = accuracy_score(teste_y,previsoes_de_base) * 100 
print("A acuracia do algoritmo de baseline foi de %.2f%%" % acuracia) 

sns.scatterplot(x="horas_esperadas",y="preco",hue=teste_y,data=teste_x)

plt.show() 
