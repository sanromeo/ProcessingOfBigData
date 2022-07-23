from pyspark.sql import Row
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([Row(Metadata='NHL 09', Link=15446),
                            Row(Metadata='Crysis 2', Link=1452),
                            Row(Metadata='FIFA Soccer 09', Link=1440),
                            Row(Metadata='FIFA Soccer 08', Link=1442),
                            Row(Metadata='Point Blank', Link=5408)
                            ])

print("-------------------TASK 2------------------------------")
print(df.show())

print("-------------------TASK 3------------------------------")
filtered = df['Link'].contains(14)
print(df.filter(filtered).show())

print("-------------------TASK 4------------------------------")
word_to_find = 'FIFA'
filtered = df['Metadata'].contains(word_to_find)
print(df.filter(filtered).select(df["Metadata"], df["Link"]).show())


