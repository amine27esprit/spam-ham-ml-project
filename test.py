import requests

BASE_URL = "http://127.0.0.1:5000"

def test_add_message():
    data = {"text": "Test message"}
    res = requests.post(f"{BASE_URL}/messages", json=data)
    print("Add message:", res.json())

def test_list_messages():
    res = requests.get(f"{BASE_URL}/messages")
    print("List messages:", res.json())

def test_update_message():
    data = {"text": "Updated message"}
    res = requests.put(f"{BASE_URL}/messages/1", json=data)
    print("Update message:", res.json())

def test_delete_message():
    res = requests.delete(f"{BASE_URL}/messages/1")
    print("Delete message:", res.json())

def test_predict():
    data = {"text": "Win a free car"}
    res = requests.post(f"{BASE_URL}/predict", json=data)
    print("Predict message:", res.json())

if __name__ == "__main__":
    print("=== TEST CRUD + IA ===")
    test_add_message()
    test_list_messages()
    test_update_message()
    test_list_messages()
    test_predict()
    test_delete_message()
    test_list_messages()
