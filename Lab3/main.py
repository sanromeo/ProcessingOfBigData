import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
import gensim.parsing.preprocessing as gsp
from gensim import utils
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline

spark = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
sc = SQLContext(spark)
DF = sc.read.csv("imdb_review.csv", header=True, inferSchema=True)
DF.show(15)
DF.printSchema()


# dividing the rating into positive (1) and negative (0)
def partition(x):
    if str(x) > str(4):
        return 1
    else:
        return 0


# replacement of the standard rating with a rating with a rating of 1 or 0 (positive or negative)
binary_udf = udf(partition, IntegerType())
DF = DF.withColumn('Rating', binary_udf('Rating'))
print(DF.show(15))


# Removing duplicates
if DF.count() > DF.dropDuplicates(['Review']).count():
    print('Data has duplicates')
print("Row count BEFORE removing duplicates:", DF.count())
DF = DF.dropDuplicates(['Review'])
print("Row count AFTER removing duplicates:", DF.count())
DF.groupBy('Rating').count().show()

# Cleaning data
filters = [
    gsp.strip_tags, gsp.strip_punctuation,
    gsp.strip_multiple_whitespaces, gsp.strip_numeric,
    gsp.remove_stopwords, gsp.strip_short, gsp.stem_text
            ]


def cleaning_text(x):
    s = x[0]
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
        return x[1], s


input_RDD = DF.rdd.map(lambda x: cleaning_text(x))
input_DataFrame = input_RDD.toDF(['Rating', 'Review'])
input_DataFrame.show(15)

# Division of data for training and testing
input_df = input_DataFrame.dropna()
train_DataFrame, test_DataFrame = input_df.randomSplit([0.8, 0.2])

# Converting text data into a matrix form (to understand the data classification model)
tokenizer = Tokenizer(inputCol="Review", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=30)
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features", minDocFreq=5)
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])
mod = pipeline.fit(train_DataFrame)
train_DataFrame = mod.transform(train_DataFrame)
test_DataFrame = mod.transform(test_DataFrame)
test_DataFrame.show(5)


# Creating and train a model
def logistic_regression(train_data, test_data):
    lr = LogisticRegression(labelCol="Rating", featuresCol="features")
    model = lr.fit(train_data)
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="Rating")
    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy of Logistic Regression Classifier : {accuracy}")
    # Creating a visual representation of the forecast
    modelSummary = model.summary
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(modelSummary.roc.select('FPR').collect(),
             modelSummary.roc.select('TPR').collect())
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


logistic_regression(train_DataFrame, test_DataFrame)
