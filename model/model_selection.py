"""
In this file I assess multiple different classifiers to assess which has the 
most potential for tuning.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import sys
path = '../src'
if path not in sys.path:
    sys.path.append(path)
from string_process import string_process
from tokenize_news import tokenize_string
import logging
logging.basicConfig(level=logging.INFO)


# The training data is in two files so I label and concatenate
def import_data():
    fake = pd.read_csv('../data/Fake.csv')
    true = pd.read_csv('../data/True.csv')
    true['class'] = 0
    fake['class'] = 1
    data = pd.concat([true, fake])
    data = data.reset_index(drop=True)
    return data

data = import_data()
data = data.sample(n=5000) # large dataset so take random sample
x = data['title']
y = data['class']


logging.info('Data succesfully imported. Training transformer pipe.')


# combine the vectorizer and transformer into a pipe for simplicity
vectorizer = CountVectorizer(preprocessor=string_process,
                             tokenizer=tokenize_string,
                             max_df=0.8,
                             max_features=2000)
tfidf = TfidfTransformer()
pipe = Pipeline([('vect', vectorizer),('tfidf', tfidf)])
vectors = pipe.fit_transform(x)


logging.info('Pipe fit to data. Training classifiers.')


# more classifiers can be added to the list for assessment
classifiers = [
    ('MultinomialNB', MultinomialNB()),
    ('SGDClassifier', SGDClassifier()),
    ('SVC', SVC())
]
x_train, x_test, y_train, y_test = train_test_split(vectors, y, test_size=0.33)
for name, clf in classifiers:
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print('Model:  '+name)
    print('Accuracy:   ' + str(accuracy_score(predictions, y_test)))
    print('\n')
    print(classification_report(predictions, y_test))
    print('='*15)
    print('\n'*3)
