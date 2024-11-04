from pyspark.sql.functions import col

def clean_data(df):
    df = df.dropDuplicates()

    df = df.na.fill(
        {
            "rating" : 0,
            "genre"  : "Unknown"
        }
    )

    df = df.filter(col("rating") > 0)

    return df