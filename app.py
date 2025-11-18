import json
from flask import Flask, request, render_template
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

with open("faq.json") as f:
    faq_data = json.load(f)

def find_answer(user_input):
    tokens = word_tokenize(user_input.lower())
    best_match = None
    best_percentage = 0  

    for item in faq_data:
        question_tokens = word_tokenize(item["question"].lower())
        matches = sum(1 for word in question_tokens if word in tokens)
        match_percentage = matches / len(question_tokens)

        print("Matched:", match_percentage, "|", item["question"])

        # Keep the best matching question
        if match_percentage > best_percentage:
            best_percentage = match_percentage
            best_match = item

    # After checking ALL questions
    if best_percentage >= 0.8:
        return best_match["answer"]
    else:
        return "Sorry, I don't understand your question."
    

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg', '')
    if not user_input:
        return "Please provide a message."
    return find_answer(user_input)

if __name__ == "__main__":
    app.run(debug=True)
