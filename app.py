import json
import random
import pickle
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf

with open("intents.json") as file:
    data = json.load(file)

model = tf.keras.models.load_model("model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

app = Flask(__name__)

def predict_class(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=10)
    prediction = model.predict(padded)[0]
    class_id = np.argmax(prediction)
    tag = label_encoder.inverse_transform([class_id])[0]
    return tag, prediction[class_id]

def find_answer(user_input):
    tag, confidence = predict_class(user_input)

    if confidence < 0.50:
        return "Sorry, I don't understand. Can you rephrase?"

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = request.args.get("msg")
    return find_answer(user_input)

if __name__ == "__main__":
    app.run(debug=True)
