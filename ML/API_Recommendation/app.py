import findspark
findspark.init()
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALSModel

app = Flask(__name__)

# Initialize Elasticsearch connection with a valid URL
es = Elasticsearch(['http://localhost:9200'])

# Initialize SparkSession
spark = SparkSession.builder.appName("RecommenderSystem").getOrCreate()

# Load the pre-trained ALS model
model = ALSModel.load('C:/Users/Youcode/Desktop/8 Months/sprint 4/sixth week_Recommandation_de_Films_Big_Data_et_IA/ML/Model/als_model')

# API endpoint to get movie recommendations
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        # Get movie title from the request
        movie_title = request.json['movie_title']
        
        # Search for movie ID in Elasticsearch
        search_body = {
            "query": {
                "match": {
                    "title": movie_title
                }
            }
        }
        response = es.search(index='moviesrating', body=search_body)
        movie_id = response['hits']['hits'][0]['_source']['movieId']
        
        # Using movie ID, search for users who rated that movie in Elasticsearch
        user_ids_query = {
            "query": {
                "term": {
                    "movieId": movie_id
                }
            },
            "_source": ["userId"]
        }
        user_ids_response = es.search(index='moviesrating', body=user_ids_query)
        user_ids = [hit['_source']['userId'] for hit in user_ids_response['hits']['hits']]
        
        # Create a Spark DataFrame with user IDs
        user_ids_df = spark.createDataFrame([(uid,) for uid in user_ids], ['userId'])
        
        # Generate movie recommendations for these users
        recommendations = model.recommendForUserSubset(user_ids_df, 3)
        recommended_movie_ids = recommendations.select('recommendations.movieId')
        
        # Fetch movie details from Elasticsearch for recommended IDs
        movie_details = []
        for row in recommended_movie_ids.collect():
            movie_id = row['movieId']
            movie_detail = es.get(index='moviesrating', id=movie_id)
            movie_details.append(movie_detail['_source'])
        
        # Return the list of detailed movie recommendations
        return jsonify({'recommendations': movie_details})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)