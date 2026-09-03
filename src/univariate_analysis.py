import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_and_split_data

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid")
COLORS = {0: '#2ca02c', 1: '#d62728'}  # Verde para Retido (0), Vermelho para Churn (1)


def calc_gaussian_pdf(x, mu, var):
    """Calcula a densidade de probabilidade (PDF) de uma Normal."""
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mu) ** 2) / (2 * var))


def analyze_continuous(X_train, y_train, feature_name, p0, p1, results_dir):
    """Realiza a análise univariada para características contínuas (Etapa 4)."""
    print(f"\n--- Analisando {feature_name} ---")

    # Extrai os dados por classe
    x0 = X_train[y_train == 0][feature_name]
    x1 = X_train[y_train == 1][feature_name]

    mu0, var0 = x0.mean(), x0.var()
    mu1, var1 = x1.mean(), x1.var()

    print(f"Classe 0 (Retido) - Média: {mu0:.2f}, Variância: {var0:.2f}")
    print(f"Classe 1 (Churn)  - Média: {mu1:.2f}, Variância: {var1:.2f}")

    # Cria um eixo x (grid) cobrindo os valores da base
    x_min, x_max = X_train[feature_name].min(), X_train[feature_name].max()
    x_grid = np.linspace(x_min, x_max, 500)

    # Etapa 2: Verossimilhança p(x|Y=c)
    pdf0 = calc_gaussian_pdf(x_grid, mu0, var0)
    pdf1 = calc_gaussian_pdf(x_grid, mu1, var1)

    # Etapa 4: Probabilidade a posteriori P(Y=c|x) = p(x|Y=c)*P(Y=c) / p(x)
    evidencia = (pdf0 * p0) + (pdf1 * p1)
    post0 = (pdf0 * p0) / evidencia
    post1 = (pdf1 * p1) / evidencia

    # Etapa 5: Fronteira de Decisão (onde P(Y=1|x) ultrapassa P(Y=0|x))
    # Procuramos o ponto no grid onde as curvas a posteriori se cruzam
    decisions = post1 > post0
    boundary_idx = np.where(np.diff(decisions))[0]

    boundaries = x_grid[boundary_idx]
    print(f"Fronteira(s) de Decisão Bayesiana encontrada(s) em: {np.round(boundaries, 2)}")

    # --- GERANDO O GRÁFICO ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot 1: Verossimilhança e Histograma
    sns.kdeplot(x0, color=COLORS[0], label='Verossimilhança Y=0', ax=axes[0], fill=True, alpha=0.3)
    sns.kdeplot(x1, color=COLORS[1], label='Verossimilhança Y=1', ax=axes[0], fill=True, alpha=0.3)
    for b in boundaries:
        axes[0].axvline(b, color='black', linestyle='--', label=f'Fronteira (x={b:.2f})')

    axes[0].set_title(f'Distribuição Condicional e Fronteira de Decisão: {feature_name}', fontsize=14)
    axes[0].set_ylabel('Densidade p(x|Y)')
    axes[0].legend()

    # Plot 2: Razão de Verossimilhança (Lambda)
    # Etapa 3: Lambda(x) = p(x|Y=1) / p(x|Y=0)
    lambda_x = pdf1 / (pdf0 + 1e-10)  # 1e-10 evita divisão por zero
    axes[1].plot(x_grid, lambda_x, color='purple', label=r'Razão $\Lambda(x)$')
    axes[1].axhline(1, color='gray', linestyle=':', label=r'$\Lambda(x) = 1$ (Empate)')

    axes[1].set_title(r'Razão de Verossimilhança $\Lambda(x) = p(x|Y=1) / p(x|Y=0)$', fontsize=12)
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel(r'$\Lambda(x)$')
    # Limita o eixo Y para não distorcer o gráfico com valores infinitos
    axes[1].set_ylim(0, max(3, np.percentile(lambda_x, 95)))
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'analise_{feature_name.replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico salvo em: {plot_path}")


def analyze_categorical(X_train, y_train, feature_name, p0, p1, results_dir):
    """Realiza a análise univariada para características categóricas (Etapa 4)."""
    print(f"\n--- Analisando {feature_name} ---")

    categories = X_train[feature_name].unique()

    # Contagens com suavização de Laplace (alpha=1)
    alpha = 1
    total0, total1 = (y_train == 0).sum(), (y_train == 1).sum()
    n_cats = len(categories)

    resultados = []

    for cat in categories:
        count0 = ((X_train[feature_name] == cat) & (y_train == 0)).sum()
        count1 = ((X_train[feature_name] == cat) & (y_train == 1)).sum()

        # Probabilidades condicionais (Verossimilhança)
        p_cat_y0 = (count0 + alpha) / (total0 + alpha * n_cats)
        p_cat_y1 = (count1 + alpha) / (total1 + alpha * n_cats)

        # Razão de Verossimilhança
        lambda_cat = p_cat_y1 / p_cat_y0

        # Regra de decisão baseada no posteriori
        post0 = p_cat_y0 * p0
        post1 = p_cat_y1 * p1
        decisao = 1 if post1 > post0 else 0

        resultados.append({
            'Categoria': cat,
            'P(x|Y=0)': p_cat_y0,
            'P(x|Y=1)': p_cat_y1,
            'Lambda': lambda_cat,
            'Decisão Bayesiana': decisao
        })

        print(f"Categoria '{cat}':")
        print(f"  P(x|Y=0) = {p_cat_y0:.4f}, P(x|Y=1) = {p_cat_y1:.4f}")
        print(f"  Razão de Verossimilhança = {lambda_cat:.4f}")
        print(f"  Decisão Bayesiana -> Y={decisao}")

    # --- GERANDO O GRÁFICO ---
    df_res = pd.DataFrame(resultados)

    x_axis = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x_axis - width / 2, df_res['P(x|Y=0)'], width, label='Verossimilhança Y=0', color=COLORS[0])
    ax.bar(x_axis + width / 2, df_res['P(x|Y=1)'], width, label='Verossimilhança Y=1', color=COLORS[1])

    ax.set_ylabel('Probabilidade P(x|Y)')
    ax.set_title(f'Distribuição Categórica: {feature_name}', fontsize=14)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(df_res['Categoria'])
    ax.legend()

    # Adiciona anotações de Lambda acima das barras com rf-string
    for i, row in enumerate(resultados):
        ax.annotate(rf"$\Lambda$ = {row['Lambda']:.2f}\nDecisão: {row['Decisão Bayesiana']}",
                    xy=(i, max(row['P(x|Y=0)'], row['P(x|Y=1)'])),
                    xytext=(0, 10), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.ylim(0, df_res[['P(x|Y=0)', 'P(x|Y=1)']].max().max() * 1.2)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'analise_{feature_name.replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico salvo em: {plot_path}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    filepath = os.path.join(project_root, 'data', 'customer_churn_dataset-training-master.csv')
    results_dir = os.path.join(project_root, 'results')

    print("Carregando dados para Análise Univariada...")
    X_train, _, y_train, _ = load_and_split_data(filepath)

    # Etapa 4: Calcula as frequências a priori das classes P(Y=0) e P(Y=1)
    total_samples = len(y_train)
    p0 = (y_train == 0).sum() / total_samples
    p1 = (y_train == 1).sum() / total_samples
    print(f"Probabilidades a priori: P(Y=0) = {p0:.4f}, P(Y=1) = {p1:.4f}")

    # Analisa cada variável isoladamente
    analyze_continuous(X_train, y_train, 'Tenure', p0, p1, results_dir)
    analyze_continuous(X_train, y_train, 'Usage Frequency', p0, p1, results_dir)
    analyze_categorical(X_train, y_train, 'Gender', p0, p1, results_dir)


if __name__ == '__main__':
    main()