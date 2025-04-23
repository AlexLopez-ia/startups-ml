"""
Funciones para visualizar resultados de modelos.
"""
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names, title="Matriz de Confusión"):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Valor real')
    plt.title(title)
    plt.tight_layout()
    plt.show()