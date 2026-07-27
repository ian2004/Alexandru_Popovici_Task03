import joblib
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


url = "https://github.com/ian2004/Alexandru_Popovici_Task03/raw/refs/heads/main/data/LinearSVC_classifier_products.pkl"

response = requests.get(url)
response.raise_for_status()

model = joblib.load(BytesIO(response.content))

while True:
    print("\n ###################### \n Modelul este pregatit! \n ######################\n")
    input_utilizator = str(input("Introduceti un item pentru a-l categoriza \n"))
    
    if input_utilizator == "exit":
        break
    else:
        prediction = model.predict([input_utilizator])
        print(prediction[0])
