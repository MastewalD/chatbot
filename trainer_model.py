import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

with open("intents.json") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=padded_sequences.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(set(training_labels)), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(padded_sequences, np.array(training_labels), epochs=500)

model.save("model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model trained successfully!")
