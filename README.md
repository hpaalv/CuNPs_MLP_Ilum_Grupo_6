# CuNPs_MLP_Ilum_Grupo_6
Trabalho de conclusão de curso da disciplina de Redes Neurais da Ilum Escola De Ciência, em que foi proposto a criação de uma rede neural MLP para algum tema relevante no meio científico.
<div align="center">
  <h1>Ilum - Escola de Ciência</h1>
</div>

<div align="center">
  <h2> Predição de Energia Total e de Formação de Nanopartículas de Cobre Utilizando Deeplearning </h2>
</div>

<div align="center">
  <h3> Grupo 6 </h3>
</div>

<div align="center">
  <h4>  
  Anna Karen Pinto;
  Beatriz Borges;
  Paulo Henrique dos Santos
  </h4>
</div>

<div align="center">
  <h5> Campinas/2024 </h5>
</div>


# Resumo  

Este trabalho visa explorar e propor um modelo de aprendizado de máquina destinado a identificar correlações entre diversas condições climáticas, incluindo temperatura, velocidade dos ventos, nível de umidade atmosférica, temperatura da superfície, precipitação e outros fatores relevantes. O objetivo principal é utilizar modelos de predição a partir das análises de dados climáticos, comparando-os, para predizer níveis de secas. 

Destaca-se que este projeto é desenvolvido como produto da disciplina de Redes Neurais, integrante do curso de Bacharelado em Ciência e Tecnologia, oferecido pela Universidade ILUM-Escola de Ciência, instituição acadêmica vinculada ao CNPEM (Centro Nacional de Pesquisa em Energia e Materiais). 

# Importando Dados 

* Baixe o arquivo intitulado "mlp_final.ipynb" desse github.
* Acesse o link: <https://data.csiro.au/collection/csiro:42598>.[1].
* Baixe a arquivo com o dataset e coloque-o na mesma parta que o arquivo "mlp_final.ipynb".
* Não é necessário remenomear o arquivo de dados.

OBS.: O dataset utilizado e a lista com todos os atributos presentes no dataset estão armazenados neste repositório em arquivos intitulados "mlp_final.ipynb" e "Cu_nanoparticle_headerlist.pdf", respectivamente.

# Introdução

Materiais em escala nanométrica exibem características distintas em comparação com materiais em escalas macrométricas, devido a uma série de fatores, incluindo a superfície exposta dos materiais. Em escalas nanométricas, a relação entre área de superfície e volume é amplificada, tornando a superfície de contato proporcionalmente maior em relação ao volume do material. Essa proporção aumentada da superfície confere propriedades aos materiais nanoestruturados. [2].

Os nanomateriais possuem uma variedade de aplicações, incluindo catálise, imagiologia por ressonância magnética e liberação controlada de fármacos. Além disso, processos de modificação superficial podem ser empregados para mitigar os efeitos citotóxicos associados a certos materiais. Essas modificações podem incluir revestimentos ou funcionalizações que tornam a interação com o ambiente biológico mais favorável, reduzindo assim os efeitos adversos. São classificadas como nanométricas partículas com dimensões tipicamente entre 1-100 nm. [2][3][4]. Portanto, é fundamental entendermos como a energia total e de formação influencia no produto final e nas características para a qual a nanopartícula será designada, permitindo a implementação de medidas preventivas e o planejamento adequado, incentivando um investimento tecnológico e científico maior nesta área.

A ciência que é utilizada para esta problemática é a Rede Neural tipo MLP (multilayer perceptron) ou, em português, perceptron multicamadas, que é uma rede neural artificial moderna de alimentação direta (feedforward). Essa rede é composta por várias camadas, incluindo uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída.[5].

A rede recebe os dados na camada de entrada com seus respectivos pesos. Cada neurônio possui uma função de ativação e um viez ao qual realizará cáculos. Durante o processo de aprendizado, os pesos de conexão na rede são ajustados após o processamento de cada dado com base na quantidade de erro na saída em comparação com o resultado esperado. O qual permite que um sistema aprenda e melhore de forma autônoma, sem ser programado explicitamente, alimentando-o com grandes quantidades de dados. [5]

Esses dados podem ser coletados de diversas formas possíveis, variando de acordo com sua finalidade e recursos para a pesquisa. Entretanto, algo em comum com todo qualquer tipo de dado é que eles são armazenados em dataset. 

Dataset é um conjunto de dados estruturados em uma tabela, contendo descrições específicas de seus atributos e arquivos significativos para o conjunto. [3] Com o conjunto de dados, é possível extrair informações necessárias para a aplicação/manipulação desejada.

Ao utilizar um dataset que combina dados de solo e clima, pode-se desenvolver modelos de Machine Learning mais robustos e precisos. Esses modelos podem capturar correlações complexas entre as variáveis ambientais e, assim, melhorar a capacidade de prever secas com antecedência. 

A escolha desse tipo de dataset permite uma abordagem multidisciplinar, integrando conhecimentos, sem perder a importância científica proposta pelo trabalho final.

Este dataset não existiria sem a disponibilização pública desses dados oferecidos pela NASA POWER Project (Projeto POWER da NASA) e pelos autores da US Drought Monitor (Monitor de Seca dos EUA). [1] [4] [5].

# Metodologia
Inicialmente, procedeu-se com a importação das bibliotecas necessárias e dos dados, os quais estavam subdivididos em conjuntos, de validação, treino e teste. Essa etapa visava preparar o terreno para uma análise exploratória a fim de compreender melhor os componentes dos dados e definir os passos subsequentes. Assim, decidiu-se utilizar apenas o subconjunto de validação devido à grande quantidade de dados disponíveis.

Durante a análise detalhada, identificou-se que o conjunto continha mais de 40 milhões de registros, caracterizando-se como bigdata. Essa natureza dos dados apresentou desafios significativos em termos de complexidade e exigiu habilidades específicas. Além disso, devido a estrutura e caracterização dos dados, um modelo de previsão utilizando séries temporais também seria possível. É importante ressaltar que o tratamento de séries temporais não estava previsto no conteúdo programático da disciplina, mas a natureza flexível do dataset permitiu a liberdade de uma interpretação fora desse contexto. Portanto, o contexto da análise temporal dos dados (datação) não foi um atributo principal da predição, contudo o atributo data foi desmembrado e utilizado como alguns dos vários atributos no modelo de predição. 

Durante o processo, também foi constatada a presença de valores nulos (NaN), os quais foram exclusivamente encontrados na coluna "score" devido à coleta de dados em intervalos de sete dias. Esses valores foram removidos. A coluna "fips" (Federal Information Processing Standards), que representa os padrões desenvolvidos pelo National Institute of Standards and Technology, foi analisada para contabilizar o número de locais distintos catalogados nos dados, com o intuito de estabelecer relações entre indicadores climáticos. Entretanto, devido a esse propósito específico, a decisão tomada foi a remoção da coluna, uma vez que a análise parcial das regiões não era um objetivo.

Além disso, procedeu-se à identificação e eliminação de valores anômalos por meio do método de detecção de outliers utilizando o desvio padrão. Este método envolve a avaliação da distância de um dado específico em relação à média geral dos dados, sendo essa distância medida em unidades de desvio padrão.

A coluna "data" foi desmembrada em três novas colunas ("day", "month" e "year"), como dito anteriormente.

A partir desse ponto, trabalhou-se com três tipos distintos de conjuntos de dados, cada um submetido a diferentes métodos de seleção de atributos:

* Primeiro dataset: Este conjunto foi submetido a uma análise de multicolinearidade (Seleção VIF - Variance Inflation Factor).
* Segundo dataset: O segundo conjunto foi redimensionado utilizando o método PCA (Principal Component Analysis), cuja principal função é verificar a quantidade de atributos que influenciam num determinado sistema.
* Terceiro dataset: O terceiro conjunto foi submetido a ambos os métodos de seleção de atributos.
  
A escolha de realizar essa separação teve como objetivo determinar a ferramenta de seleção de atributos mais eficaz para o conjunto de dados escolhido, considerando as predições a serem realizadas. Após essa fase, os conjuntos foram divididos em treino e teste, dando início ao processo de realização das predições.

### Modelos de predição utilizados

Existem vários modelos de predição disponíveis para uso. A escolha dos modelos abaixo foi baseada na premissa de avaliar como cada um reagia diante dos novos datasets. Cada modelo foi testado com os conjuntos de dados recém-criados, buscando realizar uma comparação de eficiência entre eles.

**Modelo KNN**

Em síntese, o algoritmo KNN (K-Nearest Neighbors) busca classificar cada amostra de um conjunto de dados ao avaliar sua proximidade em relação aos vizinhos mais próximos. Caso a maioria desses vizinhos pertença a uma determinada classe, a amostra em análise será classificada nessa categoria específica. Este método fundamenta-se na ideia de que objetos semelhantes tendem a estar próximos uns dos outros no espaço de características, facilitando a atribuição de rótulos com base na predominância das classes dos vizinhos mais próximos.[7].

**Árvore de Decisão**

Conforme indicado pelo próprio nome, o algoritmo cria vários pontos de decisão, representados como "nós" na árvore. Em cada nó, a decisão é tomada para seguir por um caminho específico. Esses caminhos são representados pelos "ramos". Essa estrutura fundamental de uma árvore de decisão desempenham o papel de conferenciar e indicar o direcionamento para os ramos subsequentes do fluxo de decisão.[10].

**Floresta Aleatória**  

A Floresta Aleatória, um algoritmo de aprendizagem supervisionada, cria uma "floresta" composta por uma combinação (ensemble) de árvores de decisão, frequentemente treinadas por meio do método de bagging. A essência do bagging reside na ideia central de que a fusão de modelos de aprendizado contribui para uma melhoria no resultado global.

Uma vantagem da Floresta Aleatória é sua aplicabilidade tanto em tarefas de classificação quanto em tarefas de regressão, abrangendo a maioria dos sistemas atuais de aprendizado de máquina.[11].

 *SMOTE*

O SMOTE opera com a proposta fundamental de gerar exemplos sintéticos para a classe minoritária, visando ampliar sua representação no conjunto de dados. Esse processo envolve a criação de instâncias "sintéticas" entre pares de instâncias da classe minoritária. A técnica, por sua vez, calcula a diferença entre uma instância da classe minoritária e seus vizinhos mais próximos, gerando novas instâncias ponderadas por essa diferença.

O SMOTE não é um modelo preditivo em si, mas sim uma ferramenta projetada para equilibrar a quantidade de dados no conjunto. No conjunto de dados utilizado sua aplicação combinou-se com uma segunda floresta aleatória. Esse procedimento serve para verificar se a introdução de dados sintéticos no conjunto de dados produz impactos significativos.

# Resultados e Discussões

As comparações de eficiência entre os modelos de predição foram conduzidas por meio do Score. De maneira geral, o termo "score" em modelos de classificação refere-se a métricas que avaliam o desempenho do modelo na tarefa de atribuir rótulos de classe a amostras. Essas métricas são utilizadas para quantificar quão bem o modelo está fazendo suas previsões. Quanto mais próximo de 1 é o valor da quantificação, melhor o modelo.

Ao explorar os modelos de previsão nos três conjuntos de dados distintos, pode-se ter uma compreensão da precisão de suas predições.

Conforme mencionado anteriormente, há três conjuntos de dados, cada um tratado de maneiras diferentes. Três modelos distintos foram aplicados a cada conjunto de dados para comparação, e constatou-se que, para o modelo KNN, o conjunto de dados, ao ser analisado com o SCORE, que obteve a melhor pontuação foi aquele tratado apenas com VIF, alcançando uma precisão de 0.821. Os demais conjuntos, tratados apenas com PCA, obtiveram uma pontuação de 0.814, enquanto aqueles que receberam tratamento tanto de VIF quanto de PCA registraram uma pontuação de 0.795.

Para o modelo de Árvore de Decisão, ao analisar o SCORE, o conjunto de dados que obteve a melhor pontuação foi aquele tratado apenas com VIF, alcançando uma precisão de 0.803. Os demais conjuntos, tratados apenas com PCA, registraram uma pontuação de 0.793, enquanto aqueles que receberam tratamento tanto de VIF quanto de PCA obtiveram uma pontuação de 0.787.

Para o modelo de Floresta Aleatória, ao examinar o SCORE, o conjunto de dados que apresentou o desempenho mais destacado foi aquele tratado exclusivamente com VIF, atingindo uma precisão de 0.846. Nos outros conjuntos, que foram submetidos apenas ao PCA, a pontuação foi de 0.830, enquanto aqueles que receberam tratamento tanto de VIF quanto de PCA registraram um SCORE de 0.831. 

Os modelos de Floresta Aleatória, após o tratamento com SMOTE, apresentaram resultados idênticos para os três conjuntos de dados.

# Conclusão

Os resultados obtidos ao explorar diferentes modelos de predição em conjuntos de dados tratados de maneiras distintas revelaram curiosidades sobre o desempenho desses algoritmos. Como visto, para o algoritmo KNN, o conjunto de dados tratado exclusivamente com VIF demonstrou uma precisão superior, destacando a eficácia desse método específico de tratamento. Já para o algoritmo de Árvore de Decisão e Floresta Aleatória, observou-se que o tratamento com VIF também proporcionou um desempenho superior, evidenciando a influência positiva desse método nesse contexto.

É interessante notar que, ao aplicar o SMOTE aos modelos de Floresta Aleatória, os resultados foram consistentes entre os conjuntos de dados, indicando que essa técnica de balanceamento não teve um impacto diferenciado nos conjuntos analisados.

A escolha adequada do método de tratamento de dados e do algoritmo pode ter um impacto significativo no desempenho dos modelos de Machine Learning. Cada abordagem tem suas vantagens e pode ser mais apropriada dependendo das características específicas do conjunto de dados e dos objetivos da análise. A análise cuidadosa desses resultados permite tomar decisões mais informadas na escolha e otimização de modelos para tarefas futuras.

Portanto o melhor modelo para o conjunto de dados proposto nesse trabalho é o modelo de Floresta Aleatória com dados tratados com VIF. 

# Planos futuros

* Tentar abordar o mesmo dataset, porém agora com séries temporais para entender como funciona esse método de predição e como os dados se comportam diante disso;
* Utilizar o mesmo modelo em dados brasileiros;

# Curiosidades

Ao conduzir a validação cruzada em todos os modelos, independentemente de aplicar o PCA ou a validação VIF, os hiperparâmetros otimizados pela validação cruzada permanecem consistentes. Essa consistência sugere uma estabilidade nos hiperparâmetros identificados, independentemente do método específico de tratamento de dados escolhido.
Será se isso persiste em outros datasets ou apenas no nosso?

# Referências

[1] Copper Nanoparticle Data Set. Disponível em: <https://data.csiro.au/collection/csiro:42598>. Acesso em: 09 abr. 2024.

[2] OS NANOMATERIAIS E A DESCOBERTA DE NOVOS MUNDOS NA BANCADA DO QUÍMICO  |  Manuel A. Martins e Tito Trindade - Quim. Nova, Vol. 35, No. 7, 1434-1446, 2012. Disponível em: <https://www.scielo.br/j/qn/a/P8tgywDnt7nS6tGyHdQ3BCF/>. Acesso em: 02 mai. 2024.

[3] Ojha, N. K.; Zyryanov, G. V.; Majee, A.; Charushin, V. N.; Chupakhin, O. N.; Santra, S. Copper nanoparticles as inexpensive and efficient catalyst: A valuable contribution inorganic synthesis. Coordination Chemistry Reviews 2017, 353, 1–57.11.

‌
[4] Ssekatawa K, Byarugaba DK, Angwe MK, Wampande EM, Ejobi F, Nxumalo E, Maaza M, Sackey J, Kirabira JB. Phyto-Mediated Copper Oxide Nanoparticles for Antibacterial, Antioxidant and Photocatalytic Performances. Front Bioeng Biotechnol. 2022 Feb 16;10:820218. doi: 10.3389/fbioe.2022.820218. PMID: 35252130; PMCID: PMC8889028.

‌
[5] Multilayer perceptron | Wikipedia, the free encyclopedia. Disponível em <https://en.wikipedia.org/wiki/Multilayer_perceptron#:~:text=A%20multilayer%20perceptron%20(MLP)%20is,that%20is%20not%20linearly%20separable.>. Acesso em: 29 abr. 2024.

‌
[6] Federal Information Processing Standards. Disponível em: <https://pt.wikipedia.org/wiki/Federal_Information_Processing_Standards>. Acesso em: 11 nov. 2023.

[7] "O que é e como funciona o algoritmo KNN" | Didática Tech. Disponível em: <https://didatica.tech/o-que-e-e-como-funciona-o-algoritmo-knn/>. Acesso em: 15 nov. 2023.

[8] "sklearn.neighbors.KNeighborsClassifier" | scikit-learn | Documentação. Disponível em: <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>. Acesso em: 15 nove. 2023.

[9] "sklearn.pipeline.Pipeline" | scikit-learn | Documentação. Disponível em: <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>. Acesso em: 15 nov. 2023.

[10] "Como funciona o algoritmo Árvore de Decisão" | Didática Tech. Disponível em: <https://didatica.tech/como-funciona-o-algoritmo-arvore-de-decisao/>. Acesso em: 15 nov. 2023.

[11]  "O Algoritmo da Floresta Aleatória" | Medium - Machina Sapiens. Disponível em: <https://medium.com/machina-sapiens/o-algoritmo-da-floresta-aleat%C3%B3ria-3545f6babdf8>. Acesso em: 15 nov. 2023.
