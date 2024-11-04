from pyspark.sql.functions import col, split, explode, year

def generate_features(df):
    df = df.withColumn("genre", explode(split(col("genre"), ",")))

    if "release_date" in df.columns:
        df = df.withColumn("year", year(col("release_date")))

    df = df.groupBy("year").avg("rating").alias("average_rating_per_year")

    return df