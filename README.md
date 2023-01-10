# Multi-class Text Classification

This project is intended to categorize unseen articles into 5 categories. The categories are Sport, Tech, Business, Entertainment and Politics. These data is obtained from https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv . 

This data is good enough for modelling to categorize unseen articles accurately as there is not much problems detected in the data. For this project, the trained model has high accuracy and f1 score (accuracy: around 90% and f1 score: above 84%). 
## Badges

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)



## Details of Steps

Import all packages involve in this project.

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

- Data Loading
    
    Load data by passing the URL into pd.read_csv(URL)

        URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
        df = pd.read_csv(URL)

- Data Inspection

    Review the data for verication and debugging purposes before training the model. There are 2225 articles in the data and 99 duplicated data is found.

        df.info()
        df.head()

        #To check duplicated data
        df.duplicated().sum()
        df.drop_duplicates(keep=False, inplace=True)

        #To check NaNs
        df.isna().sum()

- Features Selection

    Define the data features.
        
        category = df['category']
        text = df['text']

- Data Preprocessing

    Use Tokenizer to split the sentences into a smaller units. In this model, it will takes 5000 most common words and oov_taken is used to insert specific value when unseen word is detected. 

        num_words = 5000
        oov_token = '<OOV>'

        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(text)
        word_index = tokenizer.word_index
        print(dict(list(word_index.items())[0:10]))

        text = tokenizer.texts_to_sequences(text)

    Apply padding to make the data uniform. pad_sequences function from Keras can padding the data easily. 

        padded_text = pad_sequences(text, maxlen=200, padding='post', truncating='post')

    One Hot Encoder

        ohe = OneHotEncoder(sparse=False)
        category = ohe.fit_transform(category[::,None])

    Train-Test-Split

        padded_text = np.expand_dims(padded_text, axis=-1)
        X_train, X_test, y_train, y_test = train_test_split(padded_text, category, test_size=0.2, random_state=123)

- Model Development

    In this project, bidirectional LSTM is implemented to train the model. Bidirectional LSTM trains the input sequences in both directions forwards and backwards.

    Add a Dense layer with 5 units and activation with softmax as it is a multi-class classification problems.

        embedding_layer = 64
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(num_words, embedding_layer),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_layer)),
            tf.keras.layers.Dense(embedding_layer, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
            ])
        model.summary()

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    Save the architecture of the model in PNG file.

        plot_model(model, to_file='model.png')

    Tensorboard callback.

        LOGS_PATH = os.path.join(os.getcwd(), 'logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        tb = callbacks.TensorBoard(log_dir=LOGS_PATH)

        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=10, callbacks=[tb])

    The training process is plotted on TensorBoard.

        #Train using Google Colab 
        %load_ext tensorboard
        %tensorboard --logdir logs

- Model Analysis

    Visualize the training and validation data.

        plt.figure()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.legend(['training', 'validation'])
        plt.show()

    Predict the model.

        y_predicted = model.predict(X_test)
        y_predicted = np.argmax(y_predicted,axis=1)
        y_test = np.argmax(y_test,axis=1)

    Display the confusion matrix.

        print(classification_report(y_test, y_predicted))
        cm = confusion_matrix(y_test, y_predicted)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()

- Model Saving

    To save trained model.

        model.save('model.h5')

    To save one hot encoder model.
        
        with open('ohe.pkl', 'wb') as f:
            pickle.dump(ohe, f)

    To save tokenizer.
        
        token_json = tokenizer.to_json()
        with open('tokenizer.json', 'w') as f:
            json.dump(token_json, f)





## Model Performances

- Model analysis

![model_analysis](https://user-images.githubusercontent.com/121777112/211500438-736a93ac-1d44-4c3f-87b7-07ef7f861da9.png)

- Accuracy and f1-score

![accuracy_f1_score](https://user-images.githubusercontent.com/121777112/211500399-45197d5e-8fe8-42e3-92fd-9d719dff6e89.jpg)

- Confusion matrix

![Confusion_matrix_display](https://user-images.githubusercontent.com/121777112/211500458-148a36e1-1e13-4375-be31-2c7c2f5fbee8.jpg)

- Model training
red: train, blue: validation

 ![epoch_loss](https://user-images.githubusercontent.com/121777112/211500511-342f46ba-fc84-4f82-855e-4205ea5c3c40.jpg)

 ![epoch_accuracy](https://user-images.githubusercontent.com/121777112/211500534-dcfe0684-138b-43eb-923b-1a46a3cd7f79.jpg)

- Model architecture

 ![model_architecture](https://user-images.githubusercontent.com/121777112/211500591-41b263e1-2dc5-443f-898e-0bc548e38f84.png)

## Discussion

From this project, I found out that Bidirectional LSTM produced better model performance than unidirectional. Bidirectional LSTM combined the layers from both directions and produced a more accurate outputs. As the result, the accuracy of the trained model reached 90%.
## Acknowledgements

 - https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv
