import json
import numpy as np
import random
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("wordnet")

# Load dataset
data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hi", "Hello", "Good day", "Hey", "What's up?",
                "Good morning", "Good evening", "Is anyone there?",
                "Hi there", "Hello, how are you?", "Hey, how's it going?",
                "Greetings!", "Hi, what's up?", "Howdy", "Yo"
            ],
            "responses": [
                "Hello!", "Hi there!", "How can I assist you today?", 
                "Hey! How can I help?", "Good to see you! How can I assist?"
            ]
        },
        {
            "tag": "goodbye",
            "patterns": [
                "Bye", "See you later", "Goodbye", "Take care",
                "Catch you later", "See you soon", "I'm leaving now",
                "I have to go", "Talk to you later", "Bye bye"
            ],
            "responses": [
                "Goodbye!", "See you soon!", "Take care!", 
                "Bye! Have a great day!", "Catch you later!"
            ]
        },
        {
            "tag": "enrollment",
            "patterns": [
                "How do I enroll?", "What's the enrollment process?",
                "How can I register for classes?", "Where do I enroll?",
                "What are the requirements for enrollment?", 
                "Is online enrollment available?", "When does enrollment start?",
                "What documents do I need for enrollment?", 
                "How to enroll for senior high school?", "Can I enroll online?"
            ],
            "responses": [
                "You can enroll by visiting the school's website or the administration office.",
                "Enrollment can be done online or at the campus. Required documents include ID, birth certificate, and previous report cards.",
                "The enrollment process involves filling out a form and submitting the necessary documents.",
                "You can check the enrollment schedule on our website or contact the registrar's office."
            ]
        },
        {
            "tag": "tuition_fees",
            "patterns": [
                "How much is the tuition fee?", "What are the payment options?",
                "Are there scholarships available?", "Is there a discount for early payment?",
                "What is the mode of payment?", "Do you offer installment plans?",
                "Are there additional fees aside from tuition?", "How do I pay my tuition?",
                "Can I pay online?", "Are there financial aid options?"
            ],
            "responses": [
                "Tuition fees vary depending on the program. You can check the fee structure on our website.",
                "We offer multiple payment options including bank transfer, credit card, and cash payments at the cashier.",
                "Scholarships and financial aid are available for qualified students.",
                "Yes, installment plans are offered. Please visit the finance office for more details."
            ]
        },
        {
            "tag": "subjects",
            "patterns": [
                "What subjects are offered?", "Do you have advanced math classes?",
                "Are there electives available?", "What is the curriculum for STEM?",
                "Can I change my subjects?", "Is there a list of required subjects?",
                "What are the core subjects for senior high?", "Are there special programs for gifted students?",
                "What electives can I choose?", "Are there remedial classes available?"
            ],
            "responses": [
                "We offer a variety of subjects including core and elective classes.",
                "You can choose electives depending on your track such as STEM, ABM, HUMSS, or TVL.",
                "Advanced classes are available for Mathematics and Science.",
                "You can view the full list of subjects on our website or contact the academic office."
            ]
        },
        {
            "tag": "schedules",
            "patterns": [
                "What is the class schedule?", "Do you have morning and afternoon classes?",
                "Are classes held on weekends?", "When does the school year start?",
                "What is the school calendar?", "How long is each class period?",
                "Is there a break between classes?", "Can I change my schedule?",
                "What time does school start and end?", "Is there a summer class schedule?"
            ],
            "responses": [
                "Classes are scheduled from Monday to Friday, with morning and afternoon sessions.",
                "The school year starts in August and ends in May, following a semester system.",
                "You can check the detailed class schedules on our website or through the student portal.",
                "Schedule changes can be requested through the registrar's office."
            ]
        },
        {
            "tag": "extracurriculars",
            "patterns": [
                "What clubs can I join?", "Do you have sports teams?",
                "Are there music or art programs?", "What extracurricular activities are offered?",
                "Can I join more than one club?", "Are there leadership programs?",
                "What competitions does the school participate in?", "Do you have community service programs?",
                "Are there dance groups?", "Is there a debate team?"
            ],
            "responses": [
                "We offer a variety of extracurricular activities including sports, arts, music, and academic clubs.",
                "You can join as many clubs as your schedule allows.",
                "We have leadership and community service programs for student development.",
                "For more details, visit the student affairs office or check the club directory on our website."
            ]
        },
        {
            "tag": "uniforms",
            "patterns": [
                "Do you have a school uniform?", "Where can I buy the uniform?",
                "What is the dress code?", "Are there specific uniforms for different days?",
                "Can I wear casual clothes?", "Is there a PE uniform?",
                "Are accessories allowed?", "Can I customize my uniform?",
                "What are the rules for shoes?", "Are there guidelines for hairstyles?"
            ],
            "responses": [
                "Yes, we have a prescribed school uniform for all students.",
                "Uniforms can be purchased at the school's authorized suppliers.",
                "PE uniforms are required for Physical Education classes.",
                "You can check the complete dress code policy on our website."
            ]
        },
        {
            "tag": "requirements",
            "patterns": [
                "What are the requirements for admission?", "Do I need to take an entrance exam?",
                "Is there an interview process?", "What documents are needed for registration?",
                "Do you accept transferees?", "Are there grade requirements?",
                "What is the age requirement?", "Can I enroll if I’m from another country?",
                "Are medical records required?", "Do I need a recommendation letter?"
            ],
            "responses": [
                "Admission requirements include birth certificate, previous report cards, and ID photos.",
                "Entrance exams are required for some programs. You can check the details on our website.",
                "We accept transferees subject to evaluation of previous academic records.",
                "For international students, additional documents such as visa and passport copies are needed."
            ]
        }
    ]
}

# Preprocessing
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ["?", "!"]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build ANN Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model
model.fit(train_x, train_y, epochs=500, batch_size=5, verbose=1)
model.save("chatbot_model.h5")
print("Model training complete and saved as chatbot_model.h5")
