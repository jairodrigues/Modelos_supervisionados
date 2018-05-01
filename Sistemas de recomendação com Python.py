
# coding: utf-8

# # Sistemas de recomendação com Python
# 
# Os sistemas de recomendação geralmente dependem de conjuntos de dados maiores e, especificamente, precisam ser organizados de forma particular. Devido a isso, não teremos um projeto para acompanhar este tópico, em vez disso, teremos um processo passo a passo mais intenso na criação de um sistema de recomendação com Python com o mesmo conjunto de dados de filme.
# 
# Dois tipos mais comuns de sistemas de recomendação são  baseados em ** conteúdo colaborativo e filtragem colaborativa (CF) **.
# 
# Devido ao auto grau de complexidade desses dois modelos concentraremos em fornecer um sistema de recomendação básico, sugerindo itens que são mais parecidos com um item específico, neste caso, filmes.
# 
# Tenha em mente que este não é um verdadeiro sistema de recomendação robusto, para descrevê-lo com mais precisão, apenas diz o que os filmes / itens são mais parecidos com a escolha do seu filme.
# 
# Foram utilizados dois modelos de agrupamentos supervisionados(KNN e Kmens) no proposito de encontrar um padrão entre as avaliações de cada usuario com cada filme afim de predizer possiveis filmes que possam ser recomendados com base em um filme determinado.
# 
# Entendemos durante a execução e apresentação do projeto que por mais que modelos supervisionados como o KNN e O Kmedias possam nos entregar resultados interessantes, ainda sim estariamos no baseando em predições com base em uma possibilidade não em um resultado extato, como explicado pelo professor.
# 
# Decidimos utilizar Python e as Bibliotecas Numpy(estatistica), pandas e sklearn(machine learning), pois procuramos outras maneiras de minerar dados a não ser utilizando a plataforma R como no curso,tendo em vista que a linguagem python vem ganhando muito espaço no mercado de BigData por possuir grande variedade de bibliotecas para deep learning
# 
# 
# ## Importar bibliotecas

# In[1]:


import numpy as np
import pandas as pd


# ## Obter os dados

# In[33]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('Datasets/u.data', sep='\t', names=column_names)


# In[34]:


df.head()


# Agora vamos receber os títulos do filme:

# In[35]:


movie_titles = pd.read_csv("Datasets/Movie_Id_Titles")
movie_titles.head()


# Podemos juntá-los:

# In[36]:


df = pd.merge(df,movie_titles,on='item_id')
df.head()


# In[11]:


sns.pairplot(df,hue='rating',palette='coolwarm')


# # Análise exploratória de dados
# 
# Vamos explorar os dados um pouco e dê uma olhada em alguns dos filmes mais bem classificados.
# 
# ## Visualização Importações

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().magic(u'matplotlib inline')


# Agurpamento com classificação média e número de avaliações dos 5 melhores avaliados

# In[27]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# Agrupamento com com número total de avaliações e os 5 filmes mais avaliados (mais vistos)

# In[28]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# Vamos criar um dataframes dos filmes agrupando por titulo pegando somente a média das avaliações

# In[29]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# Agora criamos mais uma coluna com as contagem de avaliações para cada filme:

# In[30]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# Total de linhas do dataframe

# In[31]:


ratings.shape


# histogramas:

# In[32]:


plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)


# Ao analizar o grafico, notamos uma grande predominancia de filmes que foram vistos apenas por um usuario

# In[33]:


plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)


# Distribuição dos filmes por pontuação, tirando os picos que possivelmente são filmes vistos apenas uma vez, temos uma distribuição normal.

# In[34]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# Cruzando as informações de média e quantidade de avaliações, notamos um aumento na contagem das avaliações conforme o total de avaliações cresce, ou seja, os filmes mais vistos, geramente são os mais bem avaliados.

# # Modelo de Recomendação Baseado em items
# 
# Agora criamos uma matriz que tenha o ID dos usuários em um acesso e o título do filme em outro eixo. Cada célula irá consistir na classificação que o usuário deu a esse filme. Observe que haverá muitos valores de NaN, porque a maioria das pessoas não viu a maioria dos filmes.

# In[35]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# Criamos um dataframe novo com os dados em uma outra dimenção a fim de buscar a correlação da avaliação de cada usuário com 
# cada filme. Os dados com NaN são filmes que não foram avaliados pelo usuario

# Filme mais avaliado:

# In[36]:


ratings.sort_values('num of ratings',ascending=False).head(10)


# Vamos escolher dois filmes: starwars, um filme de ficção científica e Liar Liar, uma comédia.

# In[37]:


ratings.head()


# Agora vamos pegar as avaliações dos usuários para esses dois filmes:

# In[38]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()


# Podemos então usar o método corrwith() para obter correlações entre duas séries:

# In[39]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# Buscamos filmes que são similares analizando a correlação entre as avaliações de starwars e lielie usando o método corrwith. Com ele encontramos a similaridade de notas dadas por pessoas que assistiram os filmes mostrados a cima e starwars

# Vamos limpar isso removendo valores de NaN e usando um DataFrame em vez de uma série:

# In[40]:


corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# Agora, se classificarmos o quadro de dados por correlação, devemos obter os filmes mais parecidos, no entanto, notemos que obtemos alguns resultados que realmente não fazem sentido. Isso ocorre porque há muitos filmes apenas assistidos uma vez por usuários que também assistiram a star wars (foi o filme mais popular).

# In[41]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# Vamos corrigir isso, filtrando filmes com menos de 100 comentários (esse valor foi escolhido com base no histograma anterior).

# In[42]:


corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# Agora, classifique os valores e observe como os títulos têm muito mais sentido: 
# Pegamos a contagem das correlações for maior que 100 ordenando pelas correlaçoes em ordem decrescente

# In[43]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# Apartir do histograma, percebemos que o filtro é interessante apartir de 100 avaliações:

# In[44]:


plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)


# Agora o mesmo para Liar Liar:

# In[45]:


corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# In[48]:


corr_liarliar.info()


# # Modelo de Recomendação Usando KNN
# 

# Padronize as variáveis

# In[37]:


from sklearn.preprocessing import StandardScaler


# ** Criamos um objeto StandardScaler() chamado scaler. **

# In[38]:


scaler = StandardScaler()


# ** Usamos o método fit() do objeto para treinar o modelo. **

# In[39]:


df.head()


# In[41]:


scaler.fit(df.drop('title',axis=1))


# ** Usamos o método .transform () para transformar os parâmetros em uma versão padronizada. **

# In[43]:


scaled_features = scaler.transform(df.drop('title',axis=1))


# ** Convertemos os parâmetros padronizados em um DataFrame e verifique o cabeçalho desse DataFrame para garantir que a transform() funcionou. **

# In[44]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[19]:


df_feat.drop('timestamp',axis=1).head()


# # Divisão treino-teste
# 
# ** Usamos o método train_test_split para dividir seus dados em um conjunto treino e teste.**

# In[20]:


from sklearn.model_selection import train_test_split


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['rating'],
                                                    test_size=0.30)


# ** Importamos o KNeighborClassifier do scikit learn. **

# In[46]:


from sklearn.neighbors import KNeighborsClassifier


# ** Criamos uma instância do modelo KNN com n_neighbors = 1 **

# In[47]:


knn = KNeighborsClassifier(n_neighbors=1)


# ** Ajustamos este modelo KNN aos dados de treinamento. **

# In[48]:


knn.fit(X_train,y_train)


# # Previsões e avaliações
# Vamos avaliar o nosso modelo KNN!

# ** Usamos o método de previsão para prever valores usando seu modelo KNN e X_test. **

# In[49]:


pred = knn.predict(X_test)


# ** Criamos uma matriz de confusão e um relatório de classificação. **

# In[50]:


from sklearn.metrics import classification_report,confusion_matrix


# In[51]:


print(confusion_matrix(y_test,pred))


# In[52]:


print(classification_report(y_test,pred))


# # Escolhendo o valor K
# Usamos o método do cotovelo para escolher um bom valor K!
# 
# ** Criamos um loop for que treina vários modelos KNN com valores k diferentes e, em seguida, mantenha um registro do error_rate para cada um desses modelos com uma lista. Consulte o notebook se você estiver confuso nesta etapa. **

# In[53]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# ** Agora criamos o seguinte gráfico usando as informações do seu loop. **

# In[54]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[58]:


knn = KNeighborsClassifier(n_neighbors=40)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# #### Para utilizar o algoritmo através da biblioteca Sklearn KNN temos que criar uma matrix com o dataframe e apartir do filme pivot o algoritmo fornece o filme mais proximo dele

# In[59]:


test = df.drop_duplicates(['user_id','title'])
df_pivot = test.pivot(index='title',columns='user_id',values='rating').fillna(0)
df_matrix = csr_matrix(df_pivot.values)


# In[60]:


modelo_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
modelo_knn.fit(df_matrix)


# In[61]:


query_index = np.random.choice(df_pivot.shape[0])
distances, indices = modelo_knn.kneighbors(df_pivot.iloc[query_index, :].reshape(1, -1), n_neighbors = 2)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recomencação para {0}:\n'.format(df_pivot.index[query_index]))
    else:
        print('{0}: {1}, com distancia de {2}'.format(i, df_pivot.index[indices.flatten()[i]], distances.flatten()[i]))


# ## K-MEANS

# #### Conhecendo nosso dataframe

# In[64]:


df.head()


# In[65]:


df.info()


# In[66]:


df.describe()


# ## Análise exploratória de dados
# 
# 

# In[68]:


sns.set_style('whitegrid')
sns.lmplot('user_id','item_id',data=df, hue='rating',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)


# In[74]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="rating",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'user_id',bins=20,alpha=0.7)


# In[75]:


from sklearn.cluster import KMeans


# In[76]:


kmeans = KMeans(n_clusters=2)


# In[79]:


kmeans.fit(df.drop('title',axis=1))


# In[80]:


kmeans.cluster_centers_


# ## Avaliação
# 

# In[100]:


def converter(cluster):
    if cluster== 1:
        return 1
    if cluster== 2:
        return 2
    if cluster== 3:
        return 3
    if cluster== 4:
        return 4
    if cluster== 5:
        return 5


# In[101]:


df['Cluster'] = df['rating'].apply(converter)


# In[102]:


df.head()


# In[103]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))

