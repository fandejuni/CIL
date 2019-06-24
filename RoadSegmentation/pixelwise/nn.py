# -*- coding: utf-8 -

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train(X, Y, layers=[10,5,1]):

    #### Model
    model = Sequential()

    for layer in layers:
        model.add(Dense(layer, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())

    model.fit(X, Y, epochs=1, batch_size=64)

    return model

def test(model, X, Y):

    # Final evaluation of the model
    scores = model.evaluate(X, Y, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    return scores
