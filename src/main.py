# import os
# from data_processing import load_and_split_data
# from bayes_models import CustomNaiveBayes
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
#
#
# def main():
#     # Define o caminho do arquivo (certifique-se de que o CSV está na pasta 'data')
#     filepath = os.path.join('data', 'customer_churn_dataset-training-master.csv')
#
#     print("--- ETAPA 6: Separação entre Treinamento e Teste ---")
#     X_train, X_test, y_train, y_test = load_and_split_data(filepath, test_size=0.2, random_state=42)
#
#     print(f"Proporção: 80% Treino | 20% Teste")
#     print(f"Tamanho do Treino: {len(X_train)} observações")
#     print(f"Tamanho do Teste: {len(X_test)} observações")
#
#     print("\n--- ETAPA 5: Treinando o classificador Naive Bayes ---")
#     nb_classifier = CustomNaiveBayes()
#     nb_classifier.fit(X_train, y_train)
#     print("Parâmetros (Médias, Variâncias e Probabilidades) estimados com sucesso!")
#
#     print("\nRealizando predições no conjunto de teste...")
#     y_pred = nb_classifier.predict(X_test)
#
#     print("\n--- ETAPA 7: Avaliação e Matriz de Confusão ---")
#     cm = confusion_matrix(y_test, y_pred)
#
#     # Extraindo os valores para o problema binário
#     vn, fp, fn, vp = cm.ravel()
#
#     print("Matriz de Confusão:")
#     print(f"                   | Predito: 0 (Retido) | Predito: 1 (Churn) |")
#     print(f"Real: 0 (Retido)   | {vn:^19} | {fp:^18} |")
#     print(f"Real: 1 (Churn)    | {fn:^19} | {vp:^18} |")
#
#     print("\nMétricas de Desempenho:")
#     print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
#     print(f"Precisão: {precision_score(y_test, y_pred):.4f}")
#     print(f"Recall:   {recall_score(y_test, y_pred):.4f}")
#     print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
#
#
# if __name__ == '__main__':
#     main()

import os
import datetime
from data_processing import load_and_split_data
from bayes_models import CustomNaiveBayes
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def main():
    # 1. Configuração de Diretórios Dinâmicos
    # Pega o caminho absoluto da pasta onde este script (main.py) está (pasta src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Volta um nível para chegar na raiz do projeto
    project_root = os.path.dirname(current_dir)

    # Agora montamos os caminhos absolutos a partir da raiz
    filepath = os.path.join(project_root, 'data', 'customer_churn_dataset-training-master.csv')
    results_dir = os.path.join(project_root, 'results')

    os.makedirs(results_dir, exist_ok=True)

    # Gera um timestamp para versionar o arquivo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f'relatorio_metricas_v_{timestamp}.txt'
    report_path = os.path.join(results_dir, nome_arquivo)

    # Variável para acumular o texto do relatório
    relatorio_texto = []

    def registrar_log(mensagem=""):
        """Imprime no console e adiciona à lista do relatório."""
        print(mensagem)
        relatorio_texto.append(mensagem)

    # 2. Execução e Logs
    registrar_log("--- ETAPA 6: Separação entre Treinamento e Teste ---")
    X_train, X_test, y_train, y_test = load_and_split_data(filepath, test_size=0.2, random_state=42)

    registrar_log("Proporção: 80% Treino | 20% Teste")
    registrar_log("Semente aleatória: 42")
    registrar_log("Atributos: " + ", ".join(X_train.columns))
    registrar_log("Modelagem: Gaussianas nos atributos numéricos; categórica com Laplace (alpha=1) no contrato.")
    registrar_log(f"Tamanho do Treino: {len(X_train)} observações")
    registrar_log(f"Tamanho do Teste: {len(X_test)} observações\n")

    registrar_log("--- ETAPA 5: Treinando o classificador Naive Bayes ---")
    nb_classifier = CustomNaiveBayes()
    nb_classifier.fit(X_train, y_train)
    registrar_log("Parâmetros (Médias, Variâncias e Probabilidades) estimados com sucesso!\n")

    registrar_log("Realizando predições no conjunto de teste...\n")
    y_pred = nb_classifier.predict(X_test)

    registrar_log("--- ETAPA 7: Avaliação e Matriz de Confusão ---")
    cm = confusion_matrix(y_test, y_pred)
    vn, fp, fn, vp = cm.ravel()

    registrar_log("Matriz de Confusão:")
    registrar_log(f"                   | Predito: 0 (Retido) | Predito: 1 (Churn) |")
    registrar_log(f"Real: 0 (Retido)   | {vn:^19} | {fp:^18} |")
    registrar_log(f"Real: 1 (Churn)    | {fn:^19} | {vp:^18} |\n")

    registrar_log("Métricas de Desempenho:")
    registrar_log(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    registrar_log(f"Precisão: {precision_score(y_test, y_pred):.4f}")
    registrar_log(f"Recall:   {recall_score(y_test, y_pred):.4f}")
    registrar_log(f"F1-Score: {f1_score(y_test, y_pred):.4f}\n")

    # 3. Gravação do Arquivo
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(relatorio_texto))

    print(f"✅ Relatório salvo com sucesso em: {report_path}")


if __name__ == '__main__':
    main()