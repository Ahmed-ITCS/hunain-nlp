

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from sklearn.utils import shuffle

FakeNewsdf = pd.read_csv("fake.csv")

TrueNewsdf = pd.read_csv("true.csv")

FakeNewsdf['real_or_fake'] = 0
TrueNewsdf['real_or_fake'] = 1

DataSet = pd.concat([FakeNewsdf, TrueNewsdf])

DataSet = shuffle(DataSet)

DataSet.dropna(inplace=True)
DataSet.drop_duplicates(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(DataSet['text'], DataSet['real_or_fake'], test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=1.0, min_df=0.5)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

svm_model = SVC(kernel='linear')

svm_model.fit(X_train_tfidf, y_train)

y_pred = svm_model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
