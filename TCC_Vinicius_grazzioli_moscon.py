#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns


# In[2]:


#importando datasets
cesta_basica = pd.read_excel('V:\\TCC_pos\\custo_cesta_basica.xls')
combustivel = pd.read_csv('V:\\TCC_pos\\2004-2021.tsv', sep='\t')
dim_cidades = pd.read_csv('V:/TCC_pos/BRAZIL_CITIES.csv', delimiter=';')
indices = pd.read_csv('V:/TCC_pos/inflacao.csv')


# # EXPLORANDO E TRATANDO O DATASET 'cesta_basica'

# In[3]:


display(cesta_basica)


# In[4]:


#indexando a primeira linha como nome de colunas
cesta_basica.columns = cesta_basica.iloc[0]
cesta_basica = cesta_basica[1:]


# In[5]:


new_column_name = "Data"

# Obtém o nome da primeira coluna com base na posição (0 no caso da primeira coluna)
current_column_name = cesta_basica.columns[0]

# Renomeia a coluna
cesta_basica.rename(columns={current_column_name: new_column_name}, inplace=True)

#elimina as 3 ultimas linhas
cesta_basica = cesta_basica.iloc[:-3] 


# In[6]:


# Verifique quantos valores nulos existem por coluna
valores_nulos_por_coluna = cesta_basica.isnull().sum()

# Exiba o resultado
print(valores_nulos_por_coluna)


# In[7]:


# Defina o limite de valores nulos permitidos (no seu caso, 10)
limite_valores_nulos = 10

# Selecione as colunas que têm menos ou igual ao limite de valores nulos
colunas_a_manter = valores_nulos_por_coluna[valores_nulos_por_coluna <= limite_valores_nulos].index

# Crie um novo DataFrame com apenas as colunas a serem mantidas
cesta_basica_filtrada = cesta_basica[colunas_a_manter]


# In[8]:


# Usar a função melt para transformar o DataFrame
cesta_basica_melted = pd.melt(cesta_basica_filtrada, id_vars=['Data'], var_name='Cidade', value_name='Custo')

# Ordenar o DataFrame pelo nome da cidade
cesta_basica_melted = cesta_basica_melted.sort_values(by='Cidade')


# In[9]:


display(cesta_basica_melted)


# In[10]:


# Extrair os valores distintos da coluna "cidades"
cidades_distintas = cesta_basica_melted['Cidade'].unique()

# Exibir os valores distintos
print(cidades_distintas)


# # VISUALIZANDO O DATASET 'dim_cidades' E AGREGANDO AO DATASET 'cesta_basica'

# In[11]:


display(dim_cidades)


# In[12]:


total_estados = dim_cidades['STATE'].nunique()


# In[13]:


# Filtrar o DataFrame para incluir apenas as linhas com cidades da lista
dim_cidades_filtrado = dim_cidades[dim_cidades['CITY'].isin(cidades_distintas)]

# Calcular a soma da coluna 'IBGE_RES_POP' para as cidades desejadas
total_estados_estudo = dim_cidades_filtrado['STATE'].nunique()


# In[14]:


resultado = total_estados_estudo/total_estados * 100
print(resultado)


# Praticamente 60% dos estados brasileiros estão no estudo

# In[15]:


# Criar um dicionário que mapeia cidades para estados
cidade_estado_dict = dim_cidades.set_index('CITY')['STATE'].to_dict()

# Adicionar a coluna 'STATE' ao dataframe cesta_basica_melted com base no mapeamento
cesta_basica_melted['STATE'] = cesta_basica_melted['Cidade'].map(cidade_estado_dict)


# In[16]:


# Concatenar as colunas com '-' entre elas para criar um ID
cesta_basica_melted['IDs'] = cesta_basica_melted['STATE'] + '-' + cesta_basica_melted['Data'].astype(str)


# In[17]:


display(cesta_basica_melted)


# # EXPLORANDO E TRATANDO O DATASET  'combustivel'

# In[18]:


display(combustivel)


# In[19]:


print(combustivel.dtypes)


# In[20]:


colunas = combustivel.columns
print(colunas)


# In[21]:


#eliminando colunas desnecessárias
combustivel = combustivel.drop(['REGIÃO', 'PREÇO MÍNIMO REVENDA',
       'PREÇO MÁXIMO REVENDA', 'MARGEM MÉDIA REVENDA',
       'COEF DE VARIAÇÃO REVENDA', 'PREÇO MÉDIO DISTRIBUIÇÃO',
       'DESVIO PADRÃO DISTRIBUIÇÃO', 'PREÇO MÍNIMO DISTRIBUIÇÃO',
       'PREÇO MÁXIMO DISTRIBUIÇÃO', 'COEF DE VARIAÇÃO DISTRIBUIÇÃO'],axis=1)


# In[22]:


#Explorando dados de cada coluna
valores_distintos = combustivel['PRODUTO'].unique()
print(valores_distintos)
valores_distintos = combustivel['ESTADO'].unique()
print(valores_distintos)


# In[23]:


# Transformando todos tipos de diesel como um só
combustivel['PRODUTO'] = combustivel['PRODUTO'].str.replace('.*DIESEL.*', 'DIESEL', regex=True)


# In[24]:


estado_abreviacoes = {
    'DISTRITO FEDERAL': 'DF',
    'GOIAS': 'GO',
    'MATO GROSSO': 'MT',
    'MATO GROSSO DO SUL': 'MS',
    'ALAGOAS': 'AL',
    'BAHIA': 'BA',
    'CEARA': 'CE',
    'MARANHAO': 'MA',
    'PARAIBA': 'PB',
    'PERNAMBUCO': 'PE',
    'PIAUI': 'PI',
    'RIO GRANDE DO NORTE': 'RN',
    'SERGIPE': 'SE',
    'ACRE': 'AC',
    'AMAPA': 'AP',
    'AMAZONAS': 'AM',
    'PARA': 'PA',
    'RONDONIA': 'RO',
    'RORAIMA': 'RR',
    'TOCANTINS': 'TO',
    'ESPIRITO SANTO': 'ES',
    'MINAS GERAIS': 'MG',
    'RIO DE JANEIRO': 'RJ',
    'SAO PAULO': 'SP',
    'PARANA': 'PR',
    'RIO GRANDE DO SUL': 'RS',
    'SANTA CATARINA': 'SC'
}
# Adicione uma nova coluna "ESTADO_ABREVIACAO" com as abreviações dos estados
combustivel['ESTADO_ABREVIACAO'] = combustivel['ESTADO'].map(estado_abreviacoes)


# In[25]:


# Verificar os valores únicos após a substituição
valores_unicos = combustivel['PRODUTO'].unique()
print(valores_unicos)


# In[26]:


# Filtrar as linhas que contêm 'DIESEL' na coluna 'PRODUTO'
combustivel = combustivel[combustivel['PRODUTO'].str.contains('DIESEL')]

# Redefinir o índice, se desejar
combustivel.reset_index(drop=True, inplace=True)


# In[27]:


# Verificar os valores únicos após a substituição
valores_unicos = combustivel['UNIDADE DE MEDIDA'].unique()
print(valores_unicos)


# In[28]:


#dropando coluna de unidade
combustivel = combustivel.drop(['UNIDADE DE MEDIDA'],axis=1)


# In[29]:


#alterando nome das colunas
combustivel.rename(columns={'PRODUTO': 'TIPO_COMBUSTIVEL'}, inplace=True)
combustivel.rename(columns={'PREÇO MÉDIO REVENDA': 'PREÇO MÉDIO R$/L'}, inplace=True)


# In[30]:


combustivel.dtypes


# In[31]:


combustivel['DATA'] = pd.to_datetime(combustivel['DATA INICIAL']).dt.strftime('%m-%Y')

combustivel = combustivel.drop(['DATA INICIAL','DATA FINAL'],axis=1)

combustivel = combustivel.drop(['TIPO_COMBUSTIVEL'],axis=1)


# In[32]:


# Concatenar as colunas com '-' entre elas
combustivel['IDs'] = combustivel['ESTADO_ABREVIACAO'] + '-' + combustivel['DATA'].astype(str)


# In[33]:


# Exiba o DataFrame atualizado
display(combustivel)


# # UNIFICANDO DATASET combustivel + cesta_basica

# In[34]:


# Criar um dicionário que mapeia IDs para PREÇO MÉDIO R$/L
id_preco_dict = combustivel.set_index('IDs')['PREÇO MÉDIO R$/L'].to_dict()

# Adicionar a coluna 'PREÇO MÉDIO R$/L' ao dataframe cesta_basica_melted com base no mapeamento
cesta_basica_melted['PREÇO MÉDIO DIESELR$/L'] = cesta_basica_melted['IDs'].map(id_preco_dict)


# In[35]:


cesta_basica_melted = cesta_basica_melted.dropna(subset=['PREÇO MÉDIO DIESELR$/L'])
cesta_basica_melted = cesta_basica_melted.dropna(subset=['Custo'])


# In[36]:


#ordenando o dataset
cesta_basica_melted = cesta_basica_melted.sort_values(by='Data')


# In[37]:


cesta_basica_melted.rename(columns={'PREÇO MÉDIO DIESELR$/L': 'custo_diesel'}, inplace=True)


# In[38]:


cesta_basica_melted['Custo'] = pd.to_numeric(cesta_basica_melted['Custo'], errors='coerce')


# In[39]:


df_analise = cesta_basica_melted


# In[40]:


df_analise.rename(columns={'STATE': 'estado'}, inplace=True)


# In[41]:


df_analise.rename(columns={'Custo': 'valor_cb'}, inplace=True)


# In[42]:


df_analise = df_analise.drop(['IDs'],axis=1)


# In[43]:


display(df_analise)


# # TRATANDO DATASET DE INDICES ECONOMICOS BR

# In[44]:


display(indices)


# In[45]:


print(indices.dtypes)


# In[46]:


colunas_desejadas = ['ano_mes', 'ipca_acumulado_doze_meses', 'selic_ano', 'juros_reais', 'salario_minimo']
# 'ipca_variacao', 'selic_meta'
df_indices_selecionado = indices[colunas_desejadas]


# In[47]:


# Converta a coluna 'ano_mes' para o tipo datetime
df_indices_selecionado['ano_mes'] = pd.to_datetime(df_indices_selecionado['ano_mes'], format='%Y%m')

# Formate a coluna 'ano_mes' para 'YYYY-MM' e atribua-a de volta à coluna
df_indices_selecionado['ano_mes'] = df_indices_selecionado['ano_mes'].dt.strftime('%m-%Y')


# In[48]:


# Renomeie a coluna 'ano_mes' do df_indices_selecionado para 'Data' para que ela corresponda ao df_analise
df_indices_selecionado.rename(columns={'ano_mes': 'Data'}, inplace=True)


# ## Unificando datasets

# In[49]:


display(df_indices_selecionado)


# In[50]:


# Realize a mesclagem dos DataFrames com base na coluna 'Data'
df_merged = df_analise.merge(df_indices_selecionado, on='Data', how='left')


# In[69]:


df_analise = df_merged


# # Exploração dos dados

# In[70]:


display(df_analise)


# In[71]:


menor_valor = df_analise['Data'].min()

# Para encontrar o maior valor da coluna "data"
maior_valor = df_analise['Data'].max()

print(f"Menor valor da coluna 'data': {menor_valor}")
print(f"Maior valor da coluna 'data': {maior_valor}")


# In[54]:


df_analise.dtypes


# In[75]:


# Selecionar apenas as colunas do tipo float
colunas_float = df_analise.select_dtypes(include=['float'])

# Definir o número de subplots com base no número de colunas float
num_colunas = len(colunas_float.columns)

# Definir o layout dos subplots
num_linhas = num_colunas // 2 + num_colunas % 2
fig, axes = plt.subplots(num_linhas, 2, figsize=(12, 6 * num_linhas))
fig.subplots_adjust(hspace=0.8)

# Cores para os gráficos
cores = sns.color_palette('pastel', num_colunas)

# Criar gráficos para cada coluna float e imprimir resumo estatístico
for i, coluna in enumerate(colunas_float.columns):
    linha = i // 2
    coluna_subplot = i % 2
    ax = axes[linha, coluna_subplot]
    
    # Plotagem do histograma
    sns.histplot(data=colunas_float, x=coluna, bins=10, kde=True, ax=ax, color=cores[i])
    ax.set_title(f'Histograma de {coluna}')
    ax.set_xlabel('')
    
    # Imprimir o resumo estatístico abaixo do gráfico
    resumo_coluna = colunas_float[coluna].describe().reset_index()
    resumo_coluna.columns = ['Estatística', 'Valor']
    
    # Tabela de resumo
    tabela = ax.table(cellText=resumo_coluna.values, colLabels=None, cellLoc='center', loc='bottom', colColours=['lightgray', 'white'])
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    
    # Remover rótulo do eixo x
    ax.set_xlabel('')
    ax.set_xticks([])

# Exibir as informações
plt.show()


# In[56]:


correlacao = df_analise['valor_cb'].corr(df_analise['custo_diesel'])
print(f"Coeficiente de correlação: {correlacao:.2f}")


# In[77]:


sns.set(style="darkgrid")

# Gráfico de dispersão entre 'valor_cb' e 'custo_diesel'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='custo_diesel', y='valor_cb', data=df_analise, alpha=0.7, s=100, color='blue')

plt.title('Relação entre Custo da Cesta Básica e Custo do Diesel', fontsize=16)
plt.xlabel('Custo do Diesel', fontsize=14)
plt.ylabel('Custo da Cesta Básica', fontsize=14)

# Ajuste as configurações dos eixos
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Exiba o gráfico
plt.show()


# # Análise e modelos de regressão

# In[79]:


# Ajuste do modelo de regressão linear
X = df_analise['custo_diesel']  # Variável independente
y = df_analise['valor_cb']  # Variável dependente

# Adicione uma constante ao modelo (intercepto)
X = sm.add_constant(X)

# Ajuste o modelo de regressão linear
modelo = sm.OLS(y, X).fit()

# Imprima os resultados do modelo
print(modelo.summary())


# In[80]:


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Resíduos do modelo
residuals = modelo.resid

# Análise de resíduos
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Gráfico de dispersão dos resíduos em relação às variáveis independentes (X)
axes[0, 0].scatter(X['custo_diesel'], residuals)
axes[0, 0].set_title('Resíduos vs. Custo Diesel')
axes[0, 0].set_xlabel('Custo Diesel')
axes[0, 0].set_ylabel('Resíduos')

# Histograma dos resíduos
axes[0, 1].hist(residuals, bins=30)
axes[0, 1].set_title('Histograma dos Resíduos')
axes[0, 1].set_xlabel('Resíduos')
axes[0, 1].set_ylabel('Frequência')

# Q-Q plot dos resíduos para verificar a normalidade
sm.qqplot(residuals, line='s', ax=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot dos Resíduos')

# Plot dos resíduos em relação ao índice (para verificar independência)
axes[1, 1].plot(np.arange(len(residuals)), residuals)
axes[1, 1].set_title('Resíduos ao longo do Índice')
axes[1, 1].set_xlabel('Índice')
axes[1, 1].set_ylabel('Resíduos')
plt.axhline(y=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()


# Como vou começar uma analise de regressão multivariada, vou analisar a correlação linear entre as variaveis independedntes para selecionar as melhores variaveis.

# In[83]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calcule as correlações entre as variáveis independentes
correlacoes_independentes = df_analise[['valor_cb','custo_diesel', 'ipca_acumulado_doze_meses', 'juros_reais', 'salario_minimo', 'selic_ano']].corr()

# Crie um mapa de calor das correlações
plt.figure(figsize=(10, 8))
sns.heatmap(correlacoes_independentes, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor das Correlações entre Variáveis")
plt.show()


# In[81]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calcule as correlações entre as variáveis independentes
correlacoes_independentes = df_analise[['custo_diesel', 'ipca_acumulado_doze_meses', 'juros_reais', 'salario_minimo', 'selic_ano']].corr()

# Crie um mapa de calor das correlações
plt.figure(figsize=(10, 8))
sns.heatmap(correlacoes_independentes, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor das Correlações entre Variáveis Independentes")
plt.show()


# In[61]:


# Ajuste do modelo de regressão múltipla
X = df_analise[['custo_diesel','ipca_acumulado_doze_meses', 'juros_reais','selic_ano']]  # Variáveis independentes
y = df_analise['valor_cb']  # Variável dependente

# Adicione uma constante ao modelo (intercepto)
X = sm.add_constant(X)

# Ajuste o modelo de regressão múltipla
modelo_multipla = sm.OLS(y, X).fit()

# Imprima os resultados do modelo de regressão múltipla
print(modelo_multipla.summary())


# In[62]:


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Resíduos do modelo
residuals = modelo_multipla.resid

# Variáveis independentes
X = df_analise[['custo_diesel', 'ipca_acumulado_doze_meses', 'juros_reais', 'salario_minimo', 'selic_ano']]

# Análise de resíduos
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Gráficos de dispersão dos resíduos em relação às variáveis independentes
for i, col in enumerate(X.columns):
    axes[i].scatter(X[col], residuals)
    axes[i].set_title(f'Resíduos vs. {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Resíduos')

# Ajuste o layout
plt.tight_layout()
plt.show()


# Teste de normalidade dos resíduos (por exemplo, o teste de Shapiro-Wilk)
normality_test = stats.shapiro(residuals)
print("Estatística de teste:", normality_test[0])
print("Valor-p:", normality_test[1])


# Teus resíduos apresentam um baixissimo valor de p, por isso tu tem que usar um modelo robusto. Modelos robustos não querem variaveis colineares (e nem os de antes)

# # Modelo Robusto

# In[124]:


from sklearn.linear_model import RANSACRegressor

# Crie um modelo de regressão robusta com RANSAC
modelo_robusto = RANSACRegressor()

X = df_analise[['custo_diesel','ipca_acumulado_doze_meses', 'juros_reais','selic_ano']]  # Variáveis independentes
y = df_analise['valor_cb']  # Substitua pela sua variável dependente
# Ajuste o modelo aos seus dados
modelo_robusto.fit(X, y)

# Visualize os coeficientes estimados
coeficientes = modelo_robusto.estimator_.coef_
intercepto = modelo_robusto.estimator_.intercept_


# In[125]:


from sklearn.metrics import r2_score

# Faça previsões com o modelo robusto
previsoes = modelo_robusto.predict(X)

# Calcule o R²
r2 = r2_score(y, previsoes)
print("R²:", r2)


# In[108]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calcule o MSE e o MAE
mse = mean_squared_error(y, previsoes)
mae = mean_absolute_error(y, previsoes)

print("MSE:", mse)
print("MAE:", mae)


# In[109]:


import matplotlib.pyplot as plt

# Calcule os resíduos
residuos = y - previsoes

# Plote um gráfico de resíduos
plt.scatter(previsoes, residuos)
plt.xlabel("Previsões")
plt.ylabel("Resíduos")
plt.title("Gráfico de Resíduos")
plt.show()


# In[116]:


from sklearn.model_selection import cross_val_score

# Realize validação cruzada k-fold
scores = cross_val_score(modelo_robusto, X, y, cv=4)  # Altere o número de folds conforme necessário

# Exiba a pontuação média e o desvio padrão das pontuações
print("Pontuação Média da Validação Cruzada:", scores.mean())
print("Desvio Padrão das Pontuações da Validação Cruzada:", scores.std())


# Pontuação Média da Validação Cruzada (CV Score): Uma pontuação média de 0.883 indica que o modelo tem um bom desempenho médio na validação cruzada, sugerindo que ele é capaz de generalizar bem para diferentes conjuntos de dados de treinamento e teste.
# 
# Desvio Padrão das Pontuações da Validação Cruzada: O desvio padrão relativamente baixo (0.022) indica que as pontuações da validação cruzada estão consistentes, o que é um bom sinal de estabilidade no desempenho do modelo.
# 
# Erro Quadrático Médio (MSE): O MSE de 1537.04 é uma medida do erro médio quadrático das previsões em relação aos valores reais. Valores menores de MSE indicam um ajuste mais preciso do modelo aos dados. Neste caso, um MSE de 1537 é considerado aceitável, dependendo do contexto do problema.
# 
# Erro Absoluto Médio (MAE): O MAE de 28.77 é uma medida do erro médio absoluto das previsões. Ele fornece uma ideia do tamanho médio dos erros de previsão. Um MAE de 28.77 significa que, em média, as previsões do modelo têm um desvio absoluto de aproximadamente 28.77 unidades em relação aos valores reais.
# 
# Coeficiente de Determinação (R²): Um R² de 0.878 indica que o modelo é capaz de explicar aproximadamente 87.8% da variabilidade na variável dependente com base nas variáveis independentes incluídas no modelo. Isso é um bom indicativo de quão bem o modelo se ajusta aos dados.

# In[127]:


num_dobras = len(scores)

# Índices das dobras
indices_dobras = np.arange(1, num_dobras + 1)

# Crie o gráfico de barras
plt.figure(figsize=(8, 6))
plt.bar(indices_dobras, scores, color='b', alpha=0.7)
plt.xlabel('Dobras da Validação Cruzada')
plt.ylabel('Pontuação da Validação Cruzada')
plt.title('Pontuação da Validação Cruzada - Modelo RANSAC Regressor')
plt.xticks(indices_dobras)
plt.ylim([0, 1])  # Defina os limites do eixo y de acordo com sua escala
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Exiba o gráfico
plt.tight_layout()
plt.show()


# In[68]:


coeficientes = modelo_robusto.estimator_.coef_
intercepto = modelo_robusto.estimator_.intercept_

print("Intercepto:", intercepto)
for i, coef in enumerate(coeficientes):
    print(f"Coeficiente para X{i+1}:", coef)

