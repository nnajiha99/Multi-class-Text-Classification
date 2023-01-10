from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential, callbacks
from keras.utils import plot_model

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os, datetime
import numpy as np
import pickle
import json
import re

#1. Data Loading
URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
df = pd.read_csv(URL)

#2. Data Inspection
df.info()
df.head()

#To check duplicated data
df.duplicated().sum()
df.drop_duplicates(keep=False, inplace=True)

#To check NaNs
df.isna().sum()

df.category.unique()

#3. Features Selection
category = df['category']
text = df['text']
print(text[0])

#4. Data Preprocessing

# Tokenizer
num_words = 5000
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

text = tokenizer.texts_to_sequences(text)

# Padding
padded_text = pad_sequences(text, maxlen=200, padding='post', truncating='post')

# One Hot Encoder
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(category[::,None])

#Train-Test-Split
padded_text = np.expand_dims(padded_text, axis=-1)
X_train, X_test, y_train, y_test = train_test_split(padded_text, category, test_size=0.2, random_state=123)

#5. Model Development
embedding_layer = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, embedding_layer),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_layer)),
    tf.keras.layers.Dense(embedding_layer, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

plot_model(model, to_file='model.png')

# Tensorboard callback
LOGS_PATH = os.path.join(os.getcwd(), 'logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir=LOGS_PATH)

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=10, callbacks=[tb])

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs

#6. Model analysis
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'])
plt.show()

y_predicted = model.predict(X_test)
y_predicted = np.argmax(y_predicted,axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

#7. Model Saving

#To save trained model
model.save('model.h5')

#To save one hot encoder model
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

#To save tokenizer
token_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(token_json, f)