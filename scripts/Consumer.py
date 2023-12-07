import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Define the schema for the JSON data
schema = StructType([
    StructField("movie", StructType([
        StructField("genres", StringType(), True),
        StructField("movieId", StringType(), True),
        StructField("title", StringType(), True)
    ]), True),
    StructField("rating", StringType(), nullable=True),
    StructField("timestamp", StringType(), nullable=True),
    StructField("userId", StringType(), nullable=True)
])

# Create a Spark session
spark = SparkSession.builder \
    .appName("KafkaConsumer") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.4") \
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
    .getOrCreate()

# Subscribe to the Kafka topic
topic = 'rating'
df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", topic) \
        .load()

# Convert the value column from binary to string
value = df.selectExpr("CAST(value AS STRING)")

# Apply schema to the JSON data
schema = value.select(from_json(col("value"), schema).alias("value"))

# Select the individual columns
cleaned_df = schema.select("value.*")

streaming_es_query = cleaned_df.writeStream.outputMode("append").format("console").option("format", "json").start()
streaming_es_query.awaitTermination()