from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

def train_model(df):
    assembler = VectorAssembler(inputCols=["duration", "budget"], outputCol="features")
    df = assembler.transform(df).select("features", "rating")

    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(labelCol="rating")
    model = lr.fit(train_data)

    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    
    return model, {"RMSE": rmse}
