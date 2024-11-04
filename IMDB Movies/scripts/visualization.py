import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(df, model, evaluation_metrics):
    ratings = [row['rating'] for row in df.select("rating").collect()]
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings, bins=20, kde=True)
    plt.title("Distribuição de Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequência")
    plt.show()

    print("Métricas de Avaliação do Modelo:", evaluation_metrics)
