from flask import Flask, request, render_template
import joblib
import re
from nltk.corpus import stopwords

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Preprocessing function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Define the prediction route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        # Get the content from the user
        user_input = request.form['content']
        
        # Clean the input text
        clean_input = clean_text(user_input)
        
        # Vectorize the input text
        input_vector = vectorizer.transform([clean_input])
        
        # Predict using the model
        result = model.predict(input_vector)
        
        # Map prediction back to text
        prediction = 'Real' if result[0] == 1 else 'Fake'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
