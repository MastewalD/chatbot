import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open("faq.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]

# Convert text to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save the question vectors
with open("model.pkl", "wb") as f:
    pickle.dump(X, f)

print("Model trained and saved!")
