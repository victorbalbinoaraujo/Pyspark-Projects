import matplotlib.pyplot as plt
import seaborn as sns
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import count
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.clustering import KMeans


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def spark_session(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(local_path):
    try:
        return spark.read.csv(local_path, header=True, inferSchema=True)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

def preprocess_data(spark_df):
    indexer = StringIndexer(inputCols=['customer_zip_code_prefix', 'customer_city', 'customer_state'],
                            outputCols=['zip_code_index', 'city_index', 'state_index'])
    spark_df = indexer.fit(spark_df).transform(spark_df)

    encoder = OneHotEncoder(inputCols=['zip_code_index', 'city_index', 'state_index'],
                            outputCols=['zip_code_enc', 'city_enc', 'state_enc'])
    spark_df = encoder.fit(spark_df).transform(spark_df)

    assembler = VectorAssembler(inputCols=['zip_code_enc', 'city_enc', 'state_enc'],
                                outputCol='features')
    return assembler.transform(spark_df)

def kmeans(spark_df, k=5):
    kmeans = KMeans(k=k, seed=1)
    model = kmeans.fit(spark_df)
    return model.transform(spark_df)

def summarize_clusters(predictions):
    return predictions.groupBy('prediction').agg(count('customer_unique_id').alias('customer_count'))

def plot_cluster_distribution(cluster_summary):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='prediction', y='customer_count', data=cluster_summary.toPandas())
    plt.title('Distribuição dos Clusters de Clientes')
    plt.xlabel('Cluster')
    plt.ylabel('Contagem de Clientes')
    plt.show()

if __name__ == "__main__":
    spark = spark_session("Brazillian E-commerce Analysis")

    local_path = "CSV Files/olist_customers_dataset.csv"
    spark_df = load_data(local_path)

    spark_df = preprocess_data(spark_df)
    predictions = kmeans(spark_df, k=5)

    predictions.select('customer_unique_id', 'features', 'prediction').show(10)

    cluster_summary = summarize_clusters(predictions)
    cluster_summary.show()

    plot_cluster_distribution(cluster_summary)