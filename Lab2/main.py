from pyspark.sql import SparkSession


def init_spark():
    spark = SparkSession.builder.appName("BigDataLab2").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():
    spark, sc = init_spark()
    df_csv = spark.read.options(header='True', inferSchema='True', delimiter=',')\
    .csv("vgsales.csv")
    df = df_csv.selectExpr('Rank as Link', 'Name as Text')
    print('----------------TASK 3---------------------------------')
    filtered = df['Link'].contains('65')
    print(df.filter(filtered).select(df["Link"], df["Text"]).show(10))

    print('----------------TASK 4---------------------------------')
    filtered = df['Text'].contains('Mario')
    print(df.filter(filtered).show(10))


if __name__ == '__main__':
    main()



