import joblib
import pandas as pd
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


url = "https://github.com/ian2004/Alexandru_Popovici_Task03/raw/refs/heads/main/data/LinearSVC_classifier_products.pkl"

response = requests.get(url)
response.raise_for_status()

model = joblib.load(BytesIO(response.content))

def build_features(title):
    return pd.DataFrame([{
        "Product Title": title,
        "title_len_chars": len(title),
        "title_len_words": len(title.split()),
        "has_digit": int(any(ch.isdigit() for ch in title)),
        "longest_word_len": max((len(w) for w in title.split()), default=0)
    }])

while True:
    print("\n ###################### \n Modelul este pregatit! \n ######################\n")
    input_utilizator = str(input("Introduceti un item pentru a-l categoriza \n"))
    
    if input_utilizator == "exit":
        print("######################\n Iesire din model \n ######################")
        break
    else:
        X_new = build_features(input_utilizator)
        prediction = model.predict(X_new)
        print("######################", prediction[0], "\n######################")
