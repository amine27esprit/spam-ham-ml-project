import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class SpamHamModel:
    def __init__(self):
        self.pipeline = None

    def train(self, dataset_path='dataset.json'):
        # Charger le dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        texts = [d['text'] for d in data]
        labels = [d['label'] for d in data]

        # Pipeline : vectorisation + Naive Bayes
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])
        self.pipeline.fit(texts, labels)
        print("Modèle IA entraîné avec succès !")

    def predict(self, text):
        if not self.pipeline:
            raise ValueError("Le modèle n'est pas entraîné")
        return self.pipeline.predict([text])[0]

# Test rapide
if __name__ == "__main__":
    model = SpamHamModel()
    model.train()
    test_message = "Free tickets for you"
    print(f"Message: '{test_message}' -> {model.predict(test_message)}")
