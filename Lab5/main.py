from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# create an entry point into all functionality in Spark
spark = SparkSession.builder.appName('Lab5').getOrCreate()
df = spark.read.csv('heart_failure_clinical_records_dataset.csv', header=True, inferSchema=True)
df = df.withColumnRenamed('DEATH_EVENT', 'label')
df = df.select([col(c).cast("double") for c in df.columns])
df.printSchema()

# write columns as rows and rows as columns
print(pd.DataFrame(df.take(5), columns=df.columns).transpose())

# returns a new Data Frame, computes statistics such as count, min, max, mean for columns
numeric_features = [t[0] for t in df.dtypes if t[1] == 'double']
print(df.select(numeric_features).describe().toPandas().transpose())

# combine cols into a single vector column
numericCols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure',
               'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
df = assembler.transform(df)
df.show()

# encode the column of DEATH_EVENT to a column of label indexes
label_stringIdx = StringIndexer(inputCol='label', outputCol='labelIndex')
df = label_stringIdx.fit(df).transform(df)
df.show()

print(pd.DataFrame(df.take(110), columns=df.columns).transpose())

# split our dataset into training and testing data
train, test = df.randomSplit([0.7, 0.3], seed=2021)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# apply random forest classifier
rf = RandomForestClassifier(featuresCol='features', labelCol='labelIndex')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure',
                   'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'labelIndex', 'rawPrediction',
                   'prediction', 'probability').show(25)
predictions.select("labelIndex", "prediction").show(10)

# evaluate our model
evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % accuracy)
print("Test Error = %s" % (1.0 - accuracy))
