import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

class SpamHamModel:
    def __init__(self, model_path="spamham_model.pkl"):
        self.pipeline = None
        self.model_path = model_path

    def train(self, dataset_path='dataset.json', save_model=True):
        # Charger le dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset non trouvé : {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = [d['text'] for d in data]
        labels = [d['label'] for d in data]

        # Séparer training/test pour évaluer précision
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Pipeline : vectorisation + Naive Bayes
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])

        # Entraîner
        self.pipeline.fit(X_train, y_train)

        # Évaluer
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Modèle IA entraîné avec succès ! Précision sur test set : {acc:.2f}")

        # Sauvegarder le modèle
        if save_model:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.pipeline, f)
            print(f"Modèle sauvegardé dans : {self.model_path}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modèle non trouvé : {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        print("Modèle chargé avec succès !")

    def predict(self, text):
        if not self.pipeline:
            raise ValueError("Le modèle n'est pas chargé ou entraîné")
        return self.pipeline.predict([text])[0]

# Test rapide
if __name__ == "__main__":
    model = SpamHamModel()
    
    # Entraîner le modèle (ou charger si déjà sauvegardé)
    if os.path.exists(model.model_path):
        model.load_model()
    else:
        model.train()

    # Quelques tests
    test_messages = [
        "Free tickets for you",
        "Can we meet tomorrow at 2 PM?",
        "Win a $1000 gift card now",
        "Don't forget to submit the report"
    ]

    for msg in test_messages:
        print(f"Message: '{msg}' -> {model.predict(msg)}")
