# Classificador Naive Bayes - Previsão de Customer Churn

Este repositório contém a implementação do zero de um Classificador Bayesiano e de um Naive Bayes para a disciplina de Inteligência Artificial do Bacharelado em Ciência da Computação da Universidade Federal do Agreste de Pernambuco (UFAPE).

O objetivo do projeto é investigar experimentalmente o funcionamento da modelagem probabilística de características contínuas e categóricas aplicadas a um problema real de negócios.

## 👥 Dupla
* **Nicolas Gomes** ([@NicolasGomes99](https://github.com/NicolasGomes99))
* **José Portela** ([@joseportel4](https://github.com/joseportel4)))

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