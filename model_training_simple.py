# model_training_simple.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
df = pd.read_csv("news.csv")   # columns: text, label (0=fake,1=real)
df['text'] = df['text'].fillna("")

X = df['text'].values
y = df['label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple lightweight pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=200))
])

# Train
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Save the entire pipeline (very small file)
with open("simple_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved simple_model.pkl")
