import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Create Sample Dataset (instead of reading file)
data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
    'message':[
        'Hey, how are you?',
        'Win money now!!!',
        'Let’s meet tomorrow',
        'Claim your prize now',
        'Are you coming today?',
        'Free entry in contest'
    ]
}

df = pd.DataFrame(data)

# Step 2: Clean Dataset
df = df.dropna()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split Data
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Test with custom message
while True:
    msg = input("\nEnter a message: ")
    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)

    if result[0] == 1:
        print("🚨 Spam Message")
    else:
        print("✅ Not Spam")