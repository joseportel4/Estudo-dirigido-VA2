import pandas as pd
from sklearn.model_selection import train_test_split
from feature_config import FEATURES


def load_and_split_data(filepath, test_size=0.2, random_state=42):
    """
    Carrega o arquivo CSV, filtra as características exigidas e separa em treino/teste.
    """
    try:
        # Carrega a base completa
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}. Verifique se o CSV está na pasta correta.")

    # 1. Filtramos as 3 características escolhidas e a Variável Alvo
    cols_of_interest = [*FEATURES, 'Churn']
    df = df[cols_of_interest]

    # Removemos linhas vazias (NaN) para não quebrar a matemática
    df = df.dropna()

    # 2. Separamos as características (X) da variável alvo (y)
    X = df[list(FEATURES)]
    y = df['Churn']

    # 3. Divisão entre Treinamento e Teste (Exigência da Seção 6 do edital)
    # random_state=42 garante a reprodutibilidade dos resultados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test