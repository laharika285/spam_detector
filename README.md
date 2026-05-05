# spam_detector
📧 Spam Detection System

A machine learning–based spam detection project that classifies messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques.

🚀 Features
Text preprocessing (cleaning, tokenization, stopword removal)
Feature extraction using TF-IDF / Bag of Words
Multiple ML models (Naive Bayes, Logistic Regression)
Model evaluation (accuracy, precision, recall, F1-score)
Easy prediction on custom input text
🛠️ Tech Stack
Python
Scikit-learn
Pandas
NumPy
NLTK
📂 Project Structure
spam-detector/
│── data/                 # Dataset files
│── model/                # Saved model
│── spam_detector.py      # Main code
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
⚙️ Installation
git clone https://github.com/your-username/spam-detector.git
cd spam-detector
pip install -r requirements.txt
▶️ Usage
python spam_detector.py
🧠 Example Code
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Text cleaning
def clean_text(text):
    text = text.lower()
    return "".join([char for char in text if char not in string.punctuation])

data['message'] = data['message'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Feature extraction
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom input
msg = ["Congratulations! You won a free prize"]
msg_vec = vectorizer.transform(msg)
print("Spam" if model.predict(msg_vec)[0] == 1 else "Ham")
📊 Results
Achieves high accuracy on standard spam datasets
Fast and efficient for real-time predictions
🔮 Future Improvements
Add deep learning models (LSTM, BERT)
Deploy using Flask / Streamlit
Improve dataset and preprocessing
🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

📜 License

This project is licensed under the MIT License.
