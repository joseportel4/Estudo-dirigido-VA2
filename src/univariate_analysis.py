import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_and_split_data
from feature_config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

sns.set_theme(style="whitegrid")
COLORS = {0: "#2ca02c", 1: "#d62728"}
CLASS_LABELS = {0: "Retido", 1: "Churn"}


def calc_gaussian_log_pdf(x, mu, var):
    """Log da densidade Normal, com a mesma proteção usada no classificador."""
    if var == 0:
        var = 1e-6
    return -0.5 * np.log(2 * np.pi * var) - ((x - mu) ** 2) / (2 * var)


def calc_gaussian_pdf(x, mu, var):
    return np.exp(calc_gaussian_log_pdf(x, mu, var))


def analyze_continuous(X_train, y_train, feature_name, p0, p1, results_dir, log=print):
    """Analisa atributos numéricos sob a aproximação Gaussiana atual."""
    log(f"\n--- Analisando {feature_name} ---")
    x0 = X_train.loc[y_train == 0, feature_name]
    x1 = X_train.loc[y_train == 1, feature_name]
    mu0, var0 = x0.mean(), x0.var()
    mu1, var1 = x1.mean(), x1.var()
    log(f"Classe 0 (Retido) - Média: {mu0:.4f}, Variância: {var0:.4f}")
    log(f"Classe 1 (Churn)  - Média: {mu1:.4f}, Variância: {var1:.4f}")
    if feature_name == "Support Calls":
        log("Support Calls é uma contagem discreta; nesta comparação usamos aproximação Gaussiana.")

    x_min = X_train[feature_name].min()
    x_max = X_train[feature_name].max()
    x_grid = np.linspace(x_min, x_max, 1000)
    log_pdf0 = calc_gaussian_log_pdf(x_grid, mu0, var0)
    log_pdf1 = calc_gaussian_log_pdf(x_grid, mu1, var1)
    log_joint0 = log_pdf0 + np.log(p0)
    log_joint1 = log_pdf1 + np.log(p1)
    log_evidence = np.logaddexp(log_joint0, log_joint1)
    post0 = np.exp(log_joint0 - log_evidence)
    post1 = np.exp(log_joint1 - log_evidence)

    margin = log_joint1 - log_joint0
    boundary_idx = np.flatnonzero(np.diff(margin > 0))
    boundaries = [
        x_grid[i] - margin[i] * (x_grid[i + 1] - x_grid[i]) / (margin[i + 1] - margin[i])
        for i in boundary_idx
    ]
    log(f"Fronteira(s) Gaussiana(s), no intervalo observado: {np.round(boundaries, 4)}")
    log(f"Limiar de decisão: Lambda(x) > P(Y=0)/P(Y=1) = {p0 / p1:.4f} -> Churn")
    for value in np.unique(X_train[feature_name].quantile([0.1, 0.5, 0.9]).to_numpy()):
        l0 = calc_gaussian_log_pdf(value, mu0, var0) + np.log(p0)
        l1 = calc_gaussian_log_pdf(value, mu1, var1) + np.log(p1)
        posterior1 = float(np.exp(l1 - np.logaddexp(l0, l1)))
        decision = int(l1 > l0)
        log(f"Exemplo x={value:.2f}: P(Churn|x)={posterior1:.4f} -> {CLASS_LABELS[decision]}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    bins = (
        np.arange(x_min - 0.5, x_max + 1.5, 1)
        if feature_name == "Support Calls"
        else np.linspace(x_min, x_max, 36)
    )
    for c, values, log_pdf in [(0, x0, log_pdf0), (1, x1, log_pdf1)]:
        axes[0].hist(values, bins=bins, density=True, color=COLORS[c], alpha=0.25,
                     label=f"Dados observados: {CLASS_LABELS[c]}")
        axes[0].plot(x_grid, np.exp(log_pdf), color=COLORS[c], linewidth=2,
                     label=f"Gaussiana: {CLASS_LABELS[c]}")
    axes[0].set_title(f"Dados observados e aproximação Gaussiana: {feature_name}")
    axes[0].set_ylabel("Densidade")
    axes[0].legend()

    axes[1].plot(x_grid, post0, color=COLORS[0], label="P(Retido | x)")
    axes[1].plot(x_grid, post1, color=COLORS[1], label="P(Churn | x)")
    axes[1].axhline(0.5, color="gray", linestyle=":", label="Posterior = 0,5")
    axes[1].set_title("Probabilidades a posteriori, incluindo os priors do treino")
    axes[1].set_ylabel("Probabilidade")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend()

    axes[2].semilogy(x_grid, np.exp(log_pdf1 - log_pdf0), color="purple",
                     label=r"$\Lambda(x) = p(x|Y=1) / p(x|Y=0)$")
    axes[2].axhline(1, color="gray", linestyle=":", label="Verossimilhanças iguais")
    axes[2].axhline(p0 / p1, color="black", linestyle="--",
                   label=f"Limiar Bayesiano: {p0 / p1:.4f}")
    axes[2].set_title("Razão de verossimilhanças e limiar de decisão")
    axes[2].set_ylabel(r"$\Lambda(x)$ (escala log)")
    axes[2].set_xlabel(feature_name)
    axes[2].legend()

    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="#555555", linestyle="--", alpha=0.65)
    if boundaries:
        fig.suptitle("Fronteira(s): " + ", ".join(f"{b:.2f}" for b in boundaries), fontsize=12)
    fig.tight_layout()
    plot_path = os.path.join(results_dir, f'analise_{feature_name.replace(" ", "_")}.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log(f"Gráfico: {os.path.basename(plot_path)}")


def analyze_categorical(X_train, y_train, feature_name, p0, p1, results_dir, log=print):
    """Distribuição categórica com Laplace e posterior normalizado."""
    log(f"\n--- Analisando {feature_name} ---")
    categories = sorted(X_train[feature_name].unique())
    alpha = 1
    total0, total1 = (y_train == 0).sum(), (y_train == 1).sum()
    n_cats = len(categories)
    results = []
    for category in categories:
        count0 = int(((X_train[feature_name] == category) & (y_train == 0)).sum())
        count1 = int(((X_train[feature_name] == category) & (y_train == 1)).sum())
        likelihood0 = (count0 + alpha) / (total0 + alpha * n_cats)
        likelihood1 = (count1 + alpha) / (total1 + alpha * n_cats)
        joint0, joint1 = likelihood0 * p0, likelihood1 * p1
        posterior0 = joint0 / (joint0 + joint1)
        posterior1 = joint1 / (joint0 + joint1)
        decision = int(joint1 > joint0)
        ratio = likelihood1 / likelihood0
        results.append({
            "Categoria": category, "P(x|Y=0)": likelihood0, "P(x|Y=1)": likelihood1,
            "P(Y=0|x)": posterior0, "P(Y=1|x)": posterior1,
            "Lambda": ratio, "Decisao": CLASS_LABELS[decision],
            "Contagem Retido": count0,
        })
        log(f"Categoria '{category}': contagens Retido={count0}, Churn={count1}")
        log(f"  P(x|Y=0)={likelihood0:.8f}; P(x|Y=1)={likelihood1:.8f}; Lambda={ratio:.4f}")
        log(f"  P(Retido|x)={posterior0:.6f}; P(Churn|x)={posterior1:.6f} -> {CLASS_LABELS[decision]}")
        if count0 == 0 or count1 == 0:
            log("  Frequência zero nesta amostra de treino; Laplace evita probabilidade nula.")
    log("As associações descrevem esta base; não são regras universais sobre contratos.")

    frame = pd.DataFrame(results)
    positions = np.arange(len(categories))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    axes[0].bar(positions - width / 2, frame["P(x|Y=0)"], width,
                color=COLORS[0], label="P(categoria | Retido)")
    axes[0].bar(positions + width / 2, frame["P(x|Y=1)"], width,
                color=COLORS[1], label="P(categoria | Churn)")
    for i, row in enumerate(results):
        axes[0].annotate(
            f"Lambda = {row['Lambda']:.2f}\nDecisão: {row['Decisao']}",
            xy=(i, max(row["P(x|Y=0)"], row["P(x|Y=1)"])),
            xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
            fontsize=10,
        )
        if row["Contagem Retido"] == 0:
            axes[0].text(
                i, 0.70,
                "Retido: contagem observada = 0\nprobabilidade suavizada > 0",
                ha="center", va="bottom", fontsize=9, color=COLORS[0],
            )
    axes[0].set_title(f"Verossimilhanças categóricas com Laplace: {feature_name}")
    axes[0].set_ylabel("Probabilidade condicional")
    axes[0].set_ylim(0, 1)
    for c in [0, 1]:
        bars = axes[1].bar(positions + (-width / 2 if c == 0 else width / 2),
                           frame[f"P(Y={c}|x)"], width, color=COLORS[c],
                           label=f"P({CLASS_LABELS[c]} | categoria)")
        axes[1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1].axhline(0.5, color="gray", linestyle=":")
    axes[1].set_title("Probabilidades a posteriori, incluindo os priors do treino")
    axes[1].set_ylabel("Probabilidade da classe")
    axes[1].set_ylim(0, 1.2)
    axes[1].set_xlabel(feature_name)
    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(categories)
        ax.legend(loc="upper right")
    fig.tight_layout()
    plot_path = os.path.join(results_dir, f'analise_{feature_name.replace(" ", "_")}.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log(f"Gráfico: {os.path.basename(plot_path)}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(project_root, "data", "customer_churn_dataset-training-master.csv")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    report = []

    def log(message):
        print(message)
        report.append(message)

    X_train, _, y_train, _ = load_and_split_data(filepath)
    p0, p1 = (y_train == 0).mean(), (y_train == 1).mean()
    log("Análise univariada: " + ", ".join(X_train.columns))
    log(f"Somente treino: {len(y_train)} observações; divisão 80/20; random_state=42")
    log(f"Priors: P(Y=0)={p0:.6f}; P(Y=1)={p1:.6f}")
    for feature in NUMERIC_FEATURES:
        analyze_continuous(X_train, y_train, feature, p0, p1, results_dir, log)
    for feature in CATEGORICAL_FEATURES:
        analyze_categorical(X_train, y_train, feature, p0, p1, results_dir, log)
    report_path = os.path.join(results_dir, "relatorio_analise_univariada.txt")
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    print(f"Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
