# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:45:17 2019

@author: Justin
"""

import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

from data_loading import LoadData, LoadTestData, LoadTokenDict

numpy.random.seed(7)


num_words = 8000
tweet_length = 30

token_dict = LoadTokenDict("dict_full.txt", num_words)
(X_train, y_train), (X_test, y_test) = LoadData("processed/train_pos_spellchecked.txt", 
                                        "processed/train_neg_spellchecked.txt", token_dict)

embedding_vecor_length = 32

####Model
model = Sequential()
model.add(Embedding(num_words, embedding_vecor_length, input_length=tweet_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

####Fitting
model.fit(X_train, y_train, epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save("test_model.tf")


####Predicting
X = LoadTestData("processed/test_spellchecked.txt", token_dict)
y = model.predict(X)

with open("submission.csv", "w", encoding = "utf8") as file:
    file.write("Id,Prediction\n")
    for i, t in enumerate(y):
        if t>0.5:
            p = 1
        else:
            p = -1
        file.write("{},{}\n".format(i+1, p))


