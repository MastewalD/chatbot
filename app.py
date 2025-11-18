import json
import pickle
from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open("faq.json") as f:
    faq_data = json.load(f)

# Load model + vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
question_vectors = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

def find_answer(user_input):
    # Convert input into vector
    user_vector = vectorizer.transform([user_input])

    # Compute similarity
    similarities = cosine_similarity(user_vector, question_vectors).flatten()

    # Best match index
    best_index = similarities.argmax()

    # Confidence score
    score = similarities[best_index]

    if score < 0.3:
        return "Sorry, I didn't understand. Can you rephrase?"

    return faq_data[best_index]["answer"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = request.args.get("msg", "")
    return find_answer(user_input)

if __name__ == "__main__":
    app.run(debug=True)
