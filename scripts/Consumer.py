import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_date, from_unixtime, date_format, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Define the schema for the JSON data
schema = StructType([
    StructField("movie", StructType([
        StructField("genres", StringType(), True),
        StructField("movieId", StringType(), True),
        StructField("title", StringType(), True),
        StructField("release_date", StringType(), True),
    ]), True),
    StructField("rating", StringType(), nullable=True),
    StructField("timestamp", StringType(), nullable=True),
    StructField("userId", StringType(), nullable=True),
    StructField("age", StringType(), nullable=True),
    StructField("gender", StringType(), nullable=True),
    StructField("occupation", StringType(), nullable=True)
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

# Get the desired fields from the nested structure
transformed_df = cleaned_df.select(
    col("movie.*"), col("rating"), col("timestamp"), col("userId"), col("age"), col("gender"), col("occupation")
)

# Convert columns to the correct data types
transformed_df = transformed_df.withColumn("movieId", col("movieId").cast(IntegerType())) \
    .withColumn("release_date", to_date(col("release_date"), "dd-MMM-yyyy"))\
    .withColumn("rating", col("rating").cast(IntegerType())) \
    .withColumn("timestamp_date", to_date(from_unixtime(col("timestamp"))))\
    .withColumn("time", date_format(from_unixtime(col("timestamp")), "HH:mm:ss").alias("time"))\
    .withColumn("userId", col("userId").cast(IntegerType())) \
    .withColumn("age", col("age").cast(IntegerType())) \
    .withColumn(
        "time_of_day",
        when((col("time") >= "00:00:00") & (col("time") < "12:00:00"), "morning")
        .when((col("time") >= "12:00:00") & (col("time") < "17:00:00"), "afternoon")
        .when((col("time") >= "17:00:00") & (col("time") < "20:00:00"), "evening")
        .otherwise("night")
    )

# Select the desired columns for the output
final_df = transformed_df.select(
    "movieId", "title", "release_date", "genres", "rating", "timestamp_date", "time", "time_of_day", "userId", "age", "gender", "occupation"
)

# Start the streaming query
streaming_es_query = final_df.writeStream.outputMode("append").format("console").option("format", "json").start()
streaming_es_query.awaitTermination()