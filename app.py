from flask import Flask, request, jsonify
from model import SpamHamModel

app = Flask(__name__)
model = SpamHamModel()
model.train()  # Entraîner au lancement

# Stockage simple des messages
messages = []

# CRUD
@app.route('/messages', methods=['GET'])
def get_messages():
    return jsonify(messages)

@app.route('/messages', methods=['POST'])
def add_message():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "Message vide"}), 400
    message = {"id": len(messages)+1, "text": text}
    messages.append(message)
    return jsonify(message), 201

@app.route('/messages/<int:msg_id>', methods=['PUT'])
def update_message(msg_id):
    data = request.get_json()
    text = data.get('text')
    for msg in messages:
        if msg['id'] == msg_id:
            msg['text'] = text
            return jsonify(msg)
    return jsonify({"error": "Message non trouvé"}), 404

@app.route('/messages/<int:msg_id>', methods=['DELETE'])
def delete_message(msg_id):
    global messages
    messages = [m for m in messages if m['id'] != msg_id]
    return jsonify({"message": "Supprimé"}), 200

# API interne : prédiction spam/ham
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "Message vide"}), 400
    prediction = model.predict(text)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
