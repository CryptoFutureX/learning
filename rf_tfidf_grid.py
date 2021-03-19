import string
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

nlp = spacy.load('en_core_web_sm')
stopwords = list(STOP_WORDS)

punctuations = string.punctuation

# Creating a spaCy parser
parser = English()

# reading the sentiment140 data
data_path = Path.cwd() / "data" / "traindata_clean.csv"
df = pd.read_csv(data_path, encoding='latin')


def spacy_tokenizer(sentence):
    """Tokenizer"""
    tokens = parser(sentence)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    tokens = [word for word in tokens if word not in stopwords and word not in punctuations]
    return tokens


class predictors(TransformerMixin):
    """Custom transformer using spaCy """
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def clean_text(text):
    """Basic function to clean the text"""
    return text.strip().lower()


"""Parameter grid specification for grid search"""


n_estimators = [int(x) for x in np.linspace(start=10, stop=80, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [2, 4]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap
              }


"""Using GridSearchCV"""


classifier = RandomForestClassifier()
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, verbose=2, n_jobs=4)

X = df['tweet']
Y = df['target']


"""Using TF-IDF"""


tfidf_vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 2))
# Converting Pandas series to list in order to pass it to TF-IDF function
X = tfidf_vectorizer.fit_transform(X.to_list())


"""Splitting the train and test data"""


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


"""Create the  pipeline to clean, tokenize, vectorize, and classify using TF-IDF"""


grid.fit(X_train, Y_train)

print(grid.best_params_)

print(f"Train accuracy - {grid.score(X_train, Y_train):.3f} ")
print(f"Test accuracy - {grid.score(X_test, Y_test):.3f} ")

saved_model = 'rf_tfidf_grid.sav'
pickle.dump(grid, open(saved_model, 'wb'))
