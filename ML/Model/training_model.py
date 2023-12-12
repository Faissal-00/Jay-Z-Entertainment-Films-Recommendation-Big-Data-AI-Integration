# Importations nécessaires
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import os
import shutil

# Initialiser SparkSession
spark = SparkSession.builder \
    .appName("RecommenderSystem") \
    .getOrCreate()

# Définir le schéma pour le fichier u.data
schema = StructType([
    StructField("userID", IntegerType(), True),
    StructField("movieID", IntegerType(), True),
    StructField("rating", FloatType(), True)
])

# Charger les données depuis le fichier u.data
data_path = "C:/Users/Youcode/Desktop/youcode/Recommandation-de-Films-Jay-Z-Entertainment-Integration-de-Big-Data-et-IA/api/data/u.data"
ratings_df = spark.read.csv(data_path, sep='\t', header=False, schema=schema)

# Vérification 1: Afficher le schéma du DataFrame
ratings_df.printSchema()

# Vérification 2: Afficher un échantillon des données
ratings_df.show(5)

# Vérification 3: Vérifier le nombre total d'enregistrements
print(f"Nombre total d'enregistrements : {ratings_df.count()}")

# Vérification 4: Vérifier les statistiques des colonnes numériques
ratings_df.describe(["userID", "movieID", "rating"]).show()

# Ingénierie des caractéristiques : Ajouter la moyenne des évaluations par utilisateur
avg_rating_by_user = ratings_df.groupBy("userID").agg(F.avg("rating").alias("avg_rating_user"))

# Fusionner les caractéristiques avec le DataFrame des évaluations
ratings_df = ratings_df.join(avg_rating_by_user, "userID", "left")

# Diviser les données en ensembles de formation et de test
(training_data, test_data) = ratings_df.randomSplit([0.8, 0.2])

# Initialiser le modèle ALS
als = ALS(maxIter=5, regParam=0.1, userCol="userID", itemCol="movieID", ratingCol="rating")

# Entraîner le modèle sur l'ensemble de formation
model = als.fit(training_data)

# Faire des prédictions sur l'ensemble de test
predictions = model.transform(test_data)

# Vérifier les valeurs manquantes dans les prédictions avant le remplacement
print("Nombre de valeurs manquantes dans les prédictions (avant le remplacement) :")
predictions.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in predictions.columns]).show()

# Remplacer les valeurs NaN par une valeur par défaut (par exemple, 0)
predictions = predictions.na.fill(0, subset=["prediction"])

# Vérifier les valeurs manquantes dans les prédictions après le remplacement
print("Nombre de valeurs manquantes dans les prédictions (après le remplacement) :")
predictions.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in predictions.columns]).show()

# Évaluer les performances du modèle
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

# Afficher la racine carrée de l'erreur quadratique moyenne (RMSE)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Optionnel : sauvegarder le modèle pour une utilisation ultérieure
model_path = "als_model"
if os.path.exists(model_path):
    print(f"Le répertoire {model_path} existe déjà. Suppression du répertoire existant.")
    shutil.rmtree(model_path)

model.save(model_path)

# Arrêter SparkSession à la fin de la formation du modèle
spark.stop()