## Clasificare automată a produselor pe categorii
# Alexandru Popovici

# Descriere generală
Setul de date conține anunțuri de produse electrocasnice/electronice (titlu + categorie). Pornind de la coloana Product Title, modelul este antrenat să prezică Category Label folosind vectorizare TF-IDF + un clasificator liniar (LinearSVC).
Categorii disponibile (10): CPUs, Digital Cameras, Dishwashers, Freezers, Fridge Freezers, Fridges, Microwaves, Mobile Phones, TVs, Washing Machines.

# Structura proiectului
.
├── data/
│   ├── products.csv               # dataset brut, original
│   └── products_clean.csv         # dataset curățat (folosit la antrenare)
├── notebooks/
│   └── task03.ipynb               # analiză date, curățare, comparare modele
├── models/
│   └── LinearSVC_classifier_products.pkl   # modelul final, antrenat și salvat
├── train_model.py                 # antrenează și salvează modelul
├── predict_category.py            # testare interactivă: introduci un titlu, primești categoria
└── README.md

# Fluxul de lucru
Explorare date (task03.ipynb) — încărcare products.csv, verificare valori lipsă (.isna(), heatmap).
Curățare date:
    - eliminare rânduri fără Product Title sau Category Label;
    - unificare etichete duplicate/inconsistente (Mobile Phone → Mobile Phones, CPU → CPUs, fridge → Fridges);
    - eliminare coloane nefolosite pentru clasificare text (product ID, Merchant ID, _Product Code, Number_of_Views, Merchant   Rating, Listing Date);
    - salvare rezultat în products_clean.csv.
Comparare modele — s-au antrenat și evaluat 5 algoritmi pe un split train/test (80/20, stratificat), toți folosind același pipeline TfidfVectorizer + classifier:
    Model	                Acuratețe
    Logistic Regression	    0.959
    Multinomial Naive Bayes	0.938
    Decision Tree	        0.936
    Random Forest	        0.956
    LinearSVC	            0.968
**LinearSVC** a fost ales ca model final (cea mai bună acuratețe + robust pe text sparse/TF-IDF).
Antrenare model final (train_model.py) — pipeline TfidfVectorizer + LinearSVC, antrenat pe tot setul curățat, salvat cu joblib în LinearSVC_classifier_products.pkl.
Testare interactivă (predict_category.py) — încarcă modelul și cere utilizatorului un titlu de produs, apoi afișează categoria prezisă.

# Instalare & rulare

Cerințe:
    pip install pandas scikit-learn joblib requests seaborn matplotlib
Antrenarea modelului:
    python train_model.py
    - Scriptul citește products_clean.csv, antrenează pipeline-ul TF-IDF + LinearSVC și salvează modelul ca LinearSVC_classifier_products.pkl.
Testare interactivă:
    python predict_category.py
    - Vei fi întrebat să introduci titlul unui produs (ex: samsung 55 inch 4k smart tv), iar scriptul va afișa categoria prezisă. Scrie exit pentru a ieși.

# Rezultate & evaluare
Modelul final (LinearSVC) obține pe setul de test:
    - Acuratețe globală: ~96.8%
    - F1-score peste 0.90 pentru toate cele 10 categorii (majoritatea ≥ 0.95)

# Limitări cunoscute
Desii modelul prezice cu o mare acuratete majoritatea itemelor, acesta are anumite limitari in ceea ce priveste prezicerea produselor cu nume abstracte. Un test pe care l-am facut a fost sa-mi prezica categoria produselor **bosch serie 4 kgv39vl31g** si **smeg sbs8004po** care erau din categoria *Fridge Freezers*, insa modelul le-a trecut la categoria *Diswashers*.

