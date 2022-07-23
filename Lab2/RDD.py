import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *


def init_spark():
    spark = SparkSession.builder.appName("BigDataLab2").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def create_rdd(keyword):
    result = []
    for rankid, text in zip(DataFrame['Rank'], DataFrame['Name']):
        if f'{keyword}' in text:
            result.append(rankid)
    return keyword, result


DataFrame = pd.read_csv('vgsales.csv')
DataFrame.dropna(inplace=True)
spark, sc = init_spark()
keywords = sc.parallelize(
    ['Blank', 'Crysis', 'NHL', 'FIFA']
)

print("-----------------TASK 1-------------------")
print(keywords.map(create_rdd).collect())
print("-----------------TASK 1-------------------")