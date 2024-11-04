from pyspark import SparkConf
from pyspark.sql import SparkSession

conf = SparkConf()
conf.set("spark.sql.parquet.enableVectorizedReader", "false")
conf.set("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")

def load_data(csv_file):
    spark = SparkSession.builder.appName("IMDB Movies").getOrCreate()
    df = spark.read.csv(csv_file, header=True, inferSchema=True)

    return df