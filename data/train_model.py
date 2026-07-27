import pandas as pd
from sklearn.svm import LinearSVC
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

url = "https://raw.githubusercontent.com/ian2004/Alexandru_Popovici_Task03/refs/heads/main/data/products_clean.csv"
df = pd.read_csv(url)

X = df["Product Title"]
Y = df[" Category Label"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LinearSVC())
    ])

pipeline.fit(X, Y)

joblib.dump(pipeline, "LinearSVC_classifier_products.pkl")