from flask import Flask, render_template, request
import joblib
import re
from nltk.corpus import stopwords

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

# Preprocessing function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Home route (Model description, accuracy bar chart, and other content)
@app.route('/')
def home():
    return render_template('home.html')

# Dashboard route (user inputs content)
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    prediction = ""
    if request.method == 'POST':
        user_input = request.form['content']
        clean_input = clean_text(user_input)
        input_vector = vectorizer.transform([clean_input])
        result = model.predict(input_vector)
        prediction = 'Real' if result[0] == 1 else 'Fake'
    return render_template('dashboard.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
