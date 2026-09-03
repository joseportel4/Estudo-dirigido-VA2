# Classificador Naive Bayes - Previsão de Customer Churn

Este repositório contém a implementação do zero de um Classificador Bayesiano e de um Naive Bayes para a disciplina de Inteligência Artificial do Bacharelado em Ciência da Computação da Universidade Federal do Agreste de Pernambuco (UFAPE).

O objetivo do projeto é investigar experimentalmente o funcionamento da modelagem probabilística de características contínuas e categóricas aplicadas a um problema real de negócios.

## 👥 Dupla
* **Nicolas Gomes** ([@NicolasGomes99](https://github.com/NicolasGomes99))
* **José Portela** ([@joseportel4](https://github.com/joseportel4))

## 🗂️ Sobre a Base de Dados
Utilizamos o **Customer Churn Dataset**, focado em prever a rotatividade (cancelamento de serviços) de clientes. 
* **Variável Alvo:** `Churn` (1 = Cancelou, 0 = Retido).
* **Link para Download:** [Kaggle - Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

**Características Analisadas:**
1. `Tenure` (Tempo de permanência em meses) - Contínua
2. `Usage Frequency` (Dias de uso no mês) - Contínua
3. `Gender` (Gênero) - Categórica binária

*Nota: Por motivos de boas práticas de versionamento, a base de dados não está inclusa no repositório. Veja as instruções de instalação abaixo.*

## ⚙️ Pré-requisitos e Instalação

Certifique-se de ter o Python instalado em sua máquina.

1. **Clone este repositório:**
   ```bash
   git clone [https://github.com/NicolasGomes99/nome-do-repositorio.git](https://github.com/NicolasGomes99/nome-do-repositorio.git)
   cd nome-do-repositorio
   
2. **Crie e ative um ambiente virtual (Recomendado):**
   ```bash
   # No Windows
   python -m venv venv
   venv\Scripts\activate

   # No Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   
3. **Instale as dependências exigidas:**
   ```bash
   pip install -r requirements.txt
   
4. **Prepare a Base de Dados:**
   * Faça o download do dataset no link do Kaggle.
   * Crie uma pasta chamada `data` na raiz deste projeto.
   * Extraia e copie o arquivo `customer_churn_dataset-training-master.csv` para dentro da pasta `data/.`

## 💻 Comandos de Execução (Terminal)

O projeto está dividido em duas análises principais que geram artefatos automáticos na pasta `results/`. Certifique-se de estar na raiz do projeto (`nome-do-repositorio`) e com o ambiente virtual ativado antes de executar os comandos abaixo.

1. **Geração de Gráficos e Análise Univariada**
Para gerar os gráficos de distribuição, verossimilhança e calcular as fronteiras de decisão de cada característica isolada, execute:

   ```bash
   python src/univariate_analysis.py
   
* **O que acontece:** O script processa a matemática de cada variável e salva três imagens (`analise_Tenure.png`, `analise_Usage_Frequency.png` e `analise_Gender.png`) automaticamente na pasta `results/`.

2. **Geração da Matriz de Confusão e Métricas (Naive Bayes)**
Para treinar o modelo combinando as três características, realizar as predições e gerar os resultados finais do classificador, execute:

   ```bash
   python src/main.py
   
* **O que acontece:** Os resultados (Matriz de Confusão, Acurácia, Precisão, Recall e F1-Score) serão impressos no seu terminal e um arquivo de texto versionado será salvo na pasta `results/` para documentação histórica.

## 📊 Estrutura da Implementação

A modelagem segue a teoria de decisão Bayesiana, implementada da seguinte forma:

   * Hipótese de Distribuição: `Tenure` e `Usage Frequency` modeladas via distribuição Gaussiana (Normal). `Gender` modelada via distribuição de Bernoulli.


   * Teorema de Bayes: Cálculo manual das verossimilhanças $p(x|Y=c)$, razões de verossimilhança $\Lambda(x)$ e probabilidades a posteriori $P(Y=c|x)$.


   * Independência Condicional: Combinação das variáveis contínuas e categóricas assumindo independência, com aplicação no domínio logarítmico para evitar underflow numérico.


## 🎥 Vídeo de Apresentação

O vídeo contendo a explicação técnica do código, o comportamento das características e as limitações do modelo probabilístico pode ser acessado no link abaixo:

   * [Link para o Vídeo no YouTube/Drive] (Adicionar link)