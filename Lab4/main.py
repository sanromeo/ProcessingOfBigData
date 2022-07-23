from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, desc
from pyspark import SparkContext, SparkConf, SQLContext

spark = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
sc = SQLContext(spark)
data = sc.read.csv("heart_failure_clinical_records_dataset.csv", header=True, inferSchema=True)
data.show(5)

data = data.withColumnRenamed('DEATH_EVENT', 'label')
data = data.select([col(c).cast("float") for c in data.columns])
data.printSchema()

inputCols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure',
             'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
va = VectorAssembler(inputCols=inputCols, outputCol="features")
(trainingData, testData) = data.randomSplit([0.7, 0.3])

dt = DecisionTreeRegressor(featuresCol="features")
pipeline = Pipeline(stages=[va, dt])
model = pipeline.fit(trainingData)

predictions = model.transform(testData)
predictions.select("prediction", "label", "features").show(5)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

featureImportance = model.stages[1].featureImportances.toArray()
featureNames = map(lambda s: s.name, data.schema.fields)
featureImportanceMap = zip(featureImportance, featureNames)
importancesDataFrame = sc.createDataFrame(spark.parallelize(featureImportanceMap).map(lambda r: [r[1], float(r[0])]))
importancesDataFrame = importancesDataFrame.withColumnRenamed("_1", "Feature").withColumnRenamed("_2", "Importance")
importancesDataFrame.orderBy(desc("Importance")).show()
