from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import random
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import nltk

nltk.download("punkt")
nltk.download("wordnet")

app = Flask(__name__)

# Load trained model and necessary files
model = load_model("chatbot_model.h5")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
with open("intents.json") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "I'm not sure how to respond to that."
    tag = intents_list[0]["intent"]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_message = request.json.get('message')
    
    if not user_message:
        return jsonify({"response": "Error: No message provided."})

    ints = predict_class(user_message, model)
    res = get_response(ints, intents)
    return jsonify({"response": res})


if __name__ == "__main__":
    app.run(debug=True)