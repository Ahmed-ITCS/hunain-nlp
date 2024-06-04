from flask import Flask, request, render_template
import joblib

# Load the trained model and vectorizer
svm_model = joblib.load('../svm_model.pkl')
vectorizer = joblib.load('../tfidf_vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        if news_text:
            # Transform the text using the vectorizer
            text_features = vectorizer.transform([news_text])
            # Predict using the loaded SVM model
            prediction = svm_model.predict(text_features)
            result = 'Fake' if prediction[0] == 0 else 'Real'
            return render_template('index.html', prediction=result, text=news_text)
        else:
            return render_template('index.html', prediction='Please enter a news article.', text=news_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
