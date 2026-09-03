import numpy as np
import pandas as pd


class CustomNaiveBayes:
    def __init__(self):
        self.classes = []
        self.priors = {}
        self.gaussian_params = {}  # Armazena média e variância de (Tenure, Usage Frequency)
        self.categorical_params = {}  # Armazena probabilidades de (Gender)

    def fit(self, X_train, y_train):
        """
        Treina o classificador extraindo os parâmetros estatísticos do conjunto de treinamento.
        """
        self.classes = np.unique(y_train)
        n_total = len(y_train)

        cont_features = ['Tenure', 'Usage Frequency']
        cat_features = ['Gender']

        for c in self.classes:
            # Filtra os dados apenas para a classe atual (0 ou 1)
            X_c = X_train[y_train == c]

            # Etapa 4: Calcula a probabilidade a priori P(Y=c)
            self.priors[c] = len(X_c) / n_total

            # Etapa 1 e 2: Parâmetros Gaussianos para variáveis Contínuas
            for feature in cont_features:
                if feature not in self.gaussian_params:
                    self.gaussian_params[feature] = {}

                mu = X_c[feature].mean()
                var = X_c[feature].var()
                self.gaussian_params[feature][c] = (mu, var)

            # Etapa 1 e 2: Parâmetros Categóricos com Suavização de Laplace
            for feature in cat_features:
                if feature not in self.categorical_params:
                    self.categorical_params[feature] = {}

                counts = X_c[feature].value_counts().to_dict()
                total_c = len(X_c)
                unique_vals = X_train[feature].unique()
                alpha = 1  # Parâmetro de Suavização de Laplace

                self.categorical_params[feature][c] = {}
                for val in unique_vals:
                    count_val = counts.get(val, 0)
                    # P(X=x | Y=c) com Laplace
                    prob = (count_val + alpha) / (total_c + alpha * len(unique_vals))
                    self.categorical_params[feature][c][val] = prob

    def _calculate_gaussian_log_prob(self, x, mu, var):
        """
        Calcula log p(x|Y=c) para uma distribuição Normal.
        """
        if var == 0:
            var = 1e-6  # Evita divisão por zero

        term1 = -0.5 * np.log(2 * np.pi * var)
        term2 = -((x - mu) ** 2) / (2 * var)
        return term1 + term2

    def predict(self, X_test):
        """
        Etapa 5: Realiza a classificação usando as 3 características combinadas
        no domínio logarítmico para evitar underflow.
        """
        predictions = []
        cont_features = ['Tenure', 'Usage Frequency']
        cat_features = ['Gender']

        for _, row in X_test.iterrows():
            log_probs = {}
            for c in self.classes:
                # Inicializa com o log(P(Y=c))
                log_prob = np.log(self.priors[c])

                # Soma os logaritmos das verossimilhanças contínuas
                for feature in cont_features:
                    mu, var = self.gaussian_params[feature][c]
                    x_val = row[feature]
                    log_prob += self._calculate_gaussian_log_prob(x_val, mu, var)

                # Soma os logaritmos das verossimilhanças categóricas
                for feature in cat_features:
                    x_val = row[feature]
                    prob = self.categorical_params[feature][c].get(x_val, 1e-6)  # 1e-6 previne log(0)
                    log_prob += np.log(prob)

                log_probs[c] = log_prob

            # A classe prevista é a que possui o maior log probabilidade final
            best_class = max(log_probs, key=log_probs.get)
            predictions.append(best_class)

        return np.array(predictions)