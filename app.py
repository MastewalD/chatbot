import json
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open("faq.json") as f:
    faq_data = json.load(f)

# Extract all FAQ questions
questions = [item["question"] for item in faq_data]

# Initialize TF-IDF model
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def find_answer(user_input):
    # Convert user question into TF-IDF vector
    user_vector = vectorizer.transform([user_input])

    # Calculate similarity with all FAQ questions
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    # Find highest scoring FAQ question
    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]

    # Debug print (optional)
    print("Similarity score:", best_score)

    # Threshold: accept only if confidence is high enough
    if best_score >= 0.3:
        return faq_data[best_match_index]["answer"]
    else:
        return "Sorry, I didn’t understand. Can you rephrase it?"
    

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = request.args.get("msg", "")
    if not user_input:
        return "Please type something."
    return find_answer(user_input)

if __name__ == "__main__":
    app.run(debug=True)
