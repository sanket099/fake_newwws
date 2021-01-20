import pandas as pd

df = pd.read_csv('train.csv')

###Drop Nan Values
df = df.dropna()

df.head()

len(df['text'][0])

X = df.drop('label', axis=1)

y = df['label']

import tensorflow as tf
import keras
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.layers import LSTM
from keras.layers import Dense

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer

voc_size = 10000

messages = X.copy()

messages.reset_index(inplace=True)

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

onehot_repr = [one_hot(words, voc_size) for words in corpus]

sent_length = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)

from keras.layers import Dropout

## Creating model
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import numpy as np

X_final = np.array(embedded_docs)
y_final = np.array(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

y_pred = model.predict_classes(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

p = model.predict(X_test)

#y_test

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

text = ["Brace for impact? After hitting $42K, Bitcoin price volatility may rise"]
oh = [one_hot(words, voc_size) for words in text]
sent_length = 20
ed = pad_sequences(oh, padding='pre', maxlen=sent_length)
x = np.array(ed)
print(x)
mp = model.predict(x)

if (mp[0][0] >= 0.5):
    print("True")
else:
    print("Fake newssss")

print(mp)

model.save('fake_nlp.h5')

