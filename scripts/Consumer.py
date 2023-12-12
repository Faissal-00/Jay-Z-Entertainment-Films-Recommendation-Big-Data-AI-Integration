import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_date, from_unixtime, date_format, when, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Create a Spark session
spark = SparkSession.builder \
    .appName("MovieRecommandations") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.4,"
            "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0") \
    .config("es.nodes.wan.only", "true") \
    .getOrCreate()

# Define the schema for the incoming Kafka messages
schema = StructType([
    StructField("age", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("genres", StringType(), True),
    StructField("movie", StructType([
        StructField("movieId", StringType(), True),
        StructField("release_date", StringType(), True),
        StructField("title", StringType(), True)
    ]), True),
    StructField("occupation", StringType(), True),
    StructField("rating", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("userId", StringType(), True)
])

# Subscribe to the Kafka topic
topic = 'rating'
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", topic) \
    .option("startingOffsets", "earliest") \
    .load()

# Convert the value column from Kafka to a string and apply the defined schema
parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data"))

# Flatten the nested structure and select individual columns
flattened_df = parsed_df.select(
    "data.age",
    "data.gender",
    "data.movie.movieId",
    "data.movie.release_date",
    "data.movie.title",
    "data.occupation",
    "data.rating",
    "data.timestamp",
    "data.userId"
)

# Cast columns to appropriate types and apply transformations
transformed_df = flattened_df.withColumn("movieId", col("movieId").cast(IntegerType())) \
    .withColumn("release_date", to_date(col("release_date"), "dd-MMM-yyyy")) \
    .withColumn("release_date", to_date(col("release_date"))) \
    .withColumn("release_year", date_format(col("release_date"), "yyyy").cast(IntegerType())) \
    .withColumn("rating", col("rating").cast(IntegerType())) \
    .withColumn("timestamp_date", to_date(from_unixtime(col("timestamp")))) \
    .withColumn("time", date_format(from_unixtime(col("timestamp")), "HH:mm:ss")) \
    .withColumn("userId", col("userId").cast(IntegerType())) \
    .withColumn("age", col("age").cast(IntegerType()))

# Create a function to define time of day
def get_time_of_day(time):
    if time >= "00:00:00" and time < "12:00:00":
        return 'morning'
    elif time >= "12:00:00" and time < "17:00:00":
        return 'afternoon'
    elif time >= "17:00:00" and time < "20:00:00":
        return 'evening'
    else:
        return 'night'

# Apply time of day transformation
get_time_of_day_udf = udf(get_time_of_day)
transformed_df = transformed_df.withColumn("rating_in_day", get_time_of_day_udf(col("time")))

# Specify columns to drop
columns_to_drop = ['release_date', 'timestamp_date', 'time']
transformed_df = transformed_df.drop(*columns_to_drop)

# Define the desired column order
desired_columns_order = [
    'movieId', 'title', 'genres', 'release_year', 'rating', 'timestamp', 'rating_in_day',
    'userId', 'age', 'gender', 'occupation'
]

# Reorder columns in the DataFrame
transformed_df = transformed_df.select(desired_columns_order)

# Elasticsearch configuration
es_nodes = 'localhost'  # Replace with your Elasticsearch host
es_port = 9200  # Replace with your Elasticsearch port
es_resource = 'moviesrating'  # Define the Elasticsearch index and document type

# Write the DataFrame to Elasticsearch
query_write = transformed_df.writeStream \
    .outputMode("append") \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", es_nodes) \
    .option("es.port", es_port) \
    .option("es.resource", es_resource) \
    .option("checkpointLocation", "./checkpoint/data") \
    .start()

query_write.awaitTermination()