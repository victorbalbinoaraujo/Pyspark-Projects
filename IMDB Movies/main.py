from scripts.data_loading import load_data
from scripts.data_cleaning import clean_data
from scripts.feature_engineering import generate_features
from scripts.modelling import train_model
from scripts.visualization import plot_results

def main():
    data = load_data("..\\CSV Files\\imdb_movie_dataset.csv")
    
    data = clean_data(data)
    data = generate_features(data)

    model, metrics = train_model(data)

    plot_results(data, model, metrics)


if __name__ == "__main__":
    main()
