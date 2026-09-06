"""Atributos compartilhados pelo carregamento, classificador e análise."""

# Support Calls é uma contagem; mantemos a aproximação Gaussiana nesta comparação.
NUMERIC_FEATURES = ("Support Calls", "Total Spend")
CATEGORICAL_FEATURES = ("Contract Length",)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
