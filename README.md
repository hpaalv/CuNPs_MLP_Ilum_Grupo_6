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

Este trabalho visa explorar e propor um modelo de redes neurais tipo MLP destinado à predição da energia total e de formação de nanopartículas de prata, identificando quais os melhores hiperparâmetros para serem usados na rede neural que resultarão numa boa predição. Utilizando dados como número de átomos, raio máximo e mínimos das partículas para encontrar correlações entre diversas condições. O objetivo principal é utilizar modelos de predição a partir das análises de dados de nanopartículas, comparando-os, para predizer sua energia total e de formação. 

Destaca-se que este projeto é desenvolvido como produto da disciplina de Redes Neurais, integrante do curso de Bacharelado em Ciência e Tecnologia, oferecido pela Universidade Ilum - Escola de Ciência, instituição acadêmica vinculada ao CNPEM (Centro Nacional de Pesquisa em Energia e Materiais). 

# Importando Dados 

* Baixe o arquivo intitulado "mlp_final.ipynb" desse github.
* Acesse o link: <https://data.csiro.au/collection/csiro:42598>.[1].
* Baixe o arquivo com o dataset e coloque-o na mesma pasta que o arquivo "mlp_final.ipynb".
* Não é necessário renomear o arquivo de dados.

OBS.: O dataset utilizado e a lista com todos os atributos presentes no dataset estão armazenados neste repositório em arquivos intitulados "mlp_final.ipynb" e "Cu_nanoparticle_headerlist.pdf", respectivamente.

# Introdução

Materiais em escala nanométrica exibem características distintas em comparação com materiais em escalas macrométricas, devido a uma série de fatores, incluindo a superfície exposta dos materiais. Em escalas nanométricas, a relação entre área de superfície e volume é amplificada, tornando a superfície de contato proporcionalmente maior em relação ao volume do material. Essa proporção aumentada da superfície confere propriedades aos materiais nanoestruturados. [2].

Os nanomateriais possuem uma variedade de aplicações, incluindo catálise, imagiologia por ressonância magnética e liberação controlada de fármacos. Além disso, processos de modificação superficial podem ser empregados para mitigar os efeitos citotóxicos associados a certos materiais. Essas modificações podem incluir revestimentos ou funcionalizações que tornam a interação com o ambiente biológico mais favorável, reduzindo assim os efeitos adversos. São classificadas como nanométricas partículas com dimensões tipicamente entre 1-100 nm. [2][3][4]. Portanto, é fundamental entendermos como a energia total e de formação influencia no produto final e nas características para a qual a nanopartícula será designada, permitindo a implementação de medidas preventivas e o planejamento adequado, incentivando um investimento tecnológico e científico maior nesta área.

A ciência que é utilizada para esta problemática é a Rede Neural tipo MLP (multilayer perceptron) ou, em português, perceptron multicamadas, que é uma rede neural artificial moderna de alimentação direta (feedforward). Essa rede é composta por várias camadas, incluindo uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída.[5].

A rede recebe os dados na camada de entrada com seus respectivos pesos. Cada neurônio possui uma função de ativação e um viez ao qual realizará cáculos. Durante o processo de aprendizado, os pesos de conexão na rede são ajustados após o processamento de cada dado com base na quantidade de erro na saída em comparação com o resultado esperado. O qual permite que um sistema aprenda e melhore de forma autônoma, sem ser programado explicitamente, alimentando-o com grandes quantidades de dados. [5]

Esses dados podem ser coletados de diversas formas possíveis, variando de acordo com sua finalidade e recursos para a pesquisa. Entretanto, algo em comum com todo qualquer tipo de dado é que eles são armazenados em dataset. 

Dataset é um conjunto de dados estruturados em uma tabela, contendo descrições específicas de seus atributos e arquivos significativos para o conjunto. [6] Com o conjunto de dados, é possível extrair informações necessárias para a aplicação/manipulação desejada.

A escolha desse tipo de dataset permite uma abordagem multidisciplinar, integrando conhecimentos, sem perder a importância científica proposta pelo trabalho final.

# Metodologia

Inicialmente, procedeu-se com a importação das bibliotecas necessárias e dos dados, os quais foram baixados na referência [1], foram carregados em um Dataframe da biblioteca Pandas e aplicado o método "dropna" - Responsável por remover as linhas que contêm valores ausentes (NaN) do DataFrame [7] . Essa etapa visava preparar o terreno para uma análise e predição, com dados relevantes. Além disso um documento contendo os significados e as unidades de cada atributo está presente neste diretório para aqueles que desejam entender melhor os dados aplicados no projeto. 

Durante a análise detalhada, identificou-se que o conjunto continha muitos dados e grande parte deles eram valores nulos, por isso a implementação do método "dropna" foi necessária. Os dados restantes foram divididos entre features e targets, cujos targets são os valores de energia total e energia de formação das nanopartículas de cobre, e após em treino e teste. As porcentagens usadas como parâmetros para tal atividade foram definidas como 90% para treino e 10% para teste, com a semente aleatória sendo 10. A semente aleatória é um número utilizado para inicializar o gerador de números aleatórios garantindo que os resultados de operações que envolvem aleatoriedade possam ser reproduzíveis.[8]

Durante uma sessão de instrução em sala de aula, o docente sugeriu a aplicação de normalização e logaritmização nos dados, para a redução da dimensionalidade. Após a execução do procedimento dropna, onde as linhas contendo valores NaN foram eliminadas, procedeu-se à segunda etapa de logaritmização. Contudo, foi constatado que muitos dos dados continham valores nulos. O logaritmo de 0 resulta em uma indefinição matemática. Nas bibliotecas utilizadas para o cálculo do logaritmo, esse resultado é representado por um valor NaN. Consequentemente, surgiram desafios durante o treinamento da rede, devido à disparidade na quantidade de dados entre o conjunto X e Y resultando como opçõa para o grupo não realizar a logarimização.

Porém a normalização foi feita sem problema algum. A escolhida foi a normalização pelo máximo absoluto, que consiste em um método de pré-processamento de dados ao qual cada valor presente no conjunto é submetido a uma divisão pelo valor máximo encontrado. Isso resulta em uma escala onde o valor máximo é 1 e os demais valores são proporcionais a esse máximo.

O conjunto foi submetido a uma análise de multicolinearidade (Seleção VIF - Variance Inflation Factor). A multicolinearidade existe no momento em que duas ou mais variáveis independentes em um modelo de regressão múltipla apresentam alta correlação entre si. Quando algumas características são muito correlacionadas, pode-se ter dificuldade em diferenciar entre seus efeitos individuais sobre a variável dependente, ou seja, quando há multicolinearidade significativa entre as variáveis independentes em um modelo, isso pode introduzir viés nos coeficientes de regressão e afetar a interpretabilidade do modelo. O VIF funciona calculando a multicolinearidade de cada variável independente (coluna) em relação às outras variáveis independentes e com isso ele retorna um valor, quanto maior o valor do VIF para uma variável independente, maior é a multicolinearidade dessa variável com as outras variáveis independentes. [9].

O VIF implementado no código presente neste repositório foi disponibilizado pelo docente Daniel Roberto Cassar, que ministra a Disciplina de Redes Neurais e Algoritmos Genéticos da Universidade Ilum - Escola de Ciência. O seu funcionamento procede da seguinte maneira:

O algoritmo realiza uma seleção de variáveis com base no VIF. Ele recebe um array NumPy  e uma lista contendo os dados e os nomes das variáveis independentes, respectivamente. Possui também um limite máximo para o VIF. Variáveis com VIF maior que esse valor serão removidas e armazenadas numa lista. O processo segue até que todas as variáveis tenham um VIF abaixo do limite especificado. Por fim, a função retorna os dados das variáveis independentes atualizadas, a lista atualizada de nomes de variáveis e a lista de variáveis removidas.

Em seguida foi realizada a MLP [5]. O Multilayer Perceptron (MLP) é um tipo de rede neural artificial composta por várias camadas, incluindo uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída. Cada camada é formada por neurônios interconectados, cada um com sua própria função de ativação e viés.[10]

O processo de funcionamento do MLP envolve a propagação da informação, que possui pesos, através das camadas, começando pela entrada, onde os neurônios processam os dados com base em seus viés e funções de ativação. Essa informação é então transmitida para as camadas ocultas, onde o processo é repetido, podendo possuir mais de uma camada oculta, até chegar à camada de saída, que produz os resultados finais da rede neural.[10]

Um detalhe importante a se mencionar é que nossa função de ativação não é a Sigmoide, utilziada por padrão. Decidimos, por meio dos testes, utilizar a Relu, pois esta mostrou-se mais eficaz em promover a convergência mais rápida durante o treinamento, além de ajudar a mitigar problemas de desvanecimento de gradiente. A função de ativação ReLU (Rectified Linear Unit) é conhecida por sua simplicidade e eficácia, pois ativa os neurônios quando o valor de entrada é positivo e os desativa quando é negativo, facilitando o aprendizado de representações mais discriminativas nos dados. Isso pode resultar em um treinamento mais rápido e em melhores desempenhos em muitos casos, tornando-a uma escolha popular em arquiteturas de redes neurais profundas.[11]

$ 
f(x) = \begin{cases} 
x & \text{se } x > 0 \\
0 & \text{caso contrário}
\end{cases}
$

Durante o treinamento do MLP, utiliza-se o método de backpropagation para ajustar os pesos das conexões entre os neurônios. Se a saída da rede não corresponde à esperada, é calculado um erro, que é então retropropagado da camada de saída até a camada de entrada através de derivadas parciais. Os pesos das conexões são modificados de acordo com o erro propagado.[12]

O treinamento supervisionado do MLP ocorre em dois passos. Primeiro, um padrão é apresentado à camada de entrada, e a resposta é calculada até a camada de saída. Em seguida, o erro é propagado de volta para ajustar os pesos das conexões, repetindo esse processo até que o erro seja minimizado e a rede neural produza resultados precisos.[13].

Para a escolha dos hiperparâmetros, foi pensada a necessidade de uma rede que variasse a quantidade de neurônios e camadas ocultas, para que a melhor configuração possível dos dados fosse encontrada. Os hiperparâmetros foram definidos em intervalos para que a MLP pudesse variar em diferentes arquiteturas a sua estrutura, porém sempre com os mesmos dados de entrada e saída, sem alterar a quantidade de neurônios nessas camadas nas diversas conformações.

Com os hiperparâmetros definidos, um loop que inteirou de forma a criar e testar várias redes foi inserido no código, vairando a quantidade de camadas e neurônios como proposto. Para isso, criou-se duas variáveis, uma para armazenar os hiperparâmetros (a rede sem si) e outra que armazena o valor do MSE [14] (métrica adotada pelo grupo). Assim, sempre que uma nova rede fosse criada, o melhor valor do RMSE dessa rede é comparado com o armazenado na variável fixa, caso esse valor fosse menor que o armazeado, ele o substitui. Assim, a rede com o valor menor também substitui a rede já colocada na variável que armazena os hiperparâmetros.

Por fim, a rede com melhor arquitetura é selecionada e, treinada, novamente por uma quantidae maior de eras buscando minimizar ainda mais o RMSE.

# Resultados e Discussões

Após o treinamento e teste de diferentes arquiteturas de redes neurais MLP para modelagem de nanomateriais, observamos resultados promissores em termos de desempenho. Utilizamos uma variedade de arquiteturas, ajustando o número de camadas e neurônios em cada camada para encontrar a configuração mais adequada.

Ao fim dos testes, observamos consistentemente bons valores de RMSE (Root Mean Square Error), indicando que nossos modelos foram capazes de fazer previsões precisas em relação aos dados de teste.

Esses resultados validam a eficácia da abordagem de rede neural MLP para modelagem de nanomateriais e sugerem que esta técnica pode ser aplicada com sucesso em uma variedade de problemas relacionados aos materiais em escala nanométrica.

No entanto, é importante ressaltar que a avaliação dos resultados é apenas um primeiro passo na compreensão abrangente do comportamento dos nanomateriais. Pesquisas futuras podem se concentrar em explorar outras métricas de desempenho, investigar a interpretabilidade dos modelos desenvolvidos e realizar validações adicionais para garantir a robustez e generalização dos resultados.

# Conclusão

Nesta revisão, exploramos a importância dos nanomateriais e sua vasta gama de aplicações em diversas áreas, desde a catalisação até a medicina. A manipulação precisa das propriedades dos nanomateriais tem sido uma área de pesquisa em crescimento devido ao seu potencial para revolucionar tecnologias existentes e criar novas soluções para desafios atuais.

Para entender melhor e otimizar as propriedades dos nanomateriais, destacamos o uso da Rede Neural tipo MLP como uma ferramenta. A capacidade das redes neurais de aprender padrões complexos nos dados e fazer previsões precisas as torna ideais para modelar e prever o comportamento de materiais em escala nanométrica. Através do treinamento da MLP com conjuntos de dados adequados, podemos aprimorar nossa compreensão dos fatores que influenciam as propriedades dos nanomateriais e otimizar seu design para aplicações específicas.

Em suma, o estudo e a aplicação da Rede Neural MLP em conjunto com a manipulação de datasets estruturados representam uma abordagem poderosa e interdisciplinar para avançar nosso entendimento e aplicação dos nanomateriais. Esperamos que este trabalho inspire mais pesquisas e investimentos na área, levando a avanços significativos e inovações em nanotecnologia e ciência dos materiais.

# Referências

[1] Copper Nanoparticle Data Set. Disponível em: <https://data.csiro.au/collection/csiro:42598>. Acesso em: 09 abr. 2024.

[2] OS NANOMATERIAIS E A DESCOBERTA DE NOVOS MUNDOS NA BANCADA DO QUÍMICO  |  Manuel A. Martins e Tito Trindade - Quim. Nova, Vol. 35, No. 7, 1434-1446, 2012. Disponível em: <https://www.scielo.br/j/qn/a/P8tgywDnt7nS6tGyHdQ3BCF/>. Acesso em: 02 mai. 2024.

[3] Ojha, N. K.; Zyryanov, G. V.; Majee, A.; Charushin, V. N.; Chupakhin, O. N.; Santra, S. Copper nanoparticles as inexpensive and efficient catalyst: A valuable contribution inorganic synthesis. Coordination Chemistry Reviews 2017, 353, 1–57.11.

‌
[4] Ssekatawa K, Byarugaba DK, Angwe MK, Wampande EM, Ejobi F, Nxumalo E, Maaza M, Sackey J, Kirabira JB. Phyto-Mediated Copper Oxide Nanoparticles for Antibacterial, Antioxidant and Photocatalytic Performances. Front Bioeng Biotechnol. 2022 Feb 16;10:820218. doi: 10.3389/fbioe.2022.820218. PMID: 35252130; PMCID: PMC8889028.

‌
[5] Multilayer perceptron | Wikipedia, the free encyclopedia. Disponível em <https://en.wikipedia.org/wiki/Multilayer_perceptron#:~:text=A%20multilayer%20perceptron%20(MLP)%20is,that%20is%20not%20linearly%20separable.>. Acesso em: 29 abr. 2024.

[6] Dados estruturados de conjunto de dados | Central da Pesquisa Google | Documentação. Disponível em: <https://developers.google.com/search/docs/appearance/structured-data/dataset?hl=pt-br>. Acesso em: 11 nov. 2023.
‌
[7] Pandas. DataFrame.dropna | Pandas | Documentação. Disponível em: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html>. Acesso em: 02 mai. 2024.

[8] Python Random seed() Method | w3 schools. Disponível em: <https://www.w3schools.com/python/ref_random_seed.asp>. Acesso em: 02 mai. 2024.

[9] Detecting Multicollinearity with VIF – Python | geeks for geeks. Disponível em: <https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/>. Acesso em: 02 mai. 2024.

[10] Perceptron Multi-Camadas (MLP) | icmc usp. Disponível em: <https://sites.icmc.usp.br/andre/research/neural/MLP.htm>. Acesso em: 03 mai. 2024.

[11] Relu | Função de ativação:
https://www.deeplearningbook.com.br/funcao-deativacao/#:~:text=ReLU%20é%20a%20função%20de,neurônios%20ativados%20pela%20função%20ReLU. Acesso em: 03 mai. 2024.

[11] Métricas para Regressão: Entendendo as métricas R², MAE, MAPE, MSE e RMSE | medium. Disponível em: <https://medium.com/data-hackers/prevendo-n%C3%BAmeros-entendendo-m%C3%A9tricas-de-regress%C3%A3o-35545e011e70>. Acesso em: 03 mai. 2024.

