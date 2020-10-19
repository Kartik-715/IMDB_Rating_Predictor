from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from keras.layers import Bidirectional, GlobalMaxPool1D, GRU, Embedding
from keras.initializers import Constant
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression

''' Pass Data and the functions will return the trained model '''

def ApplyLogisticRegression(Reviews, Ratings):
    lr = LogisticRegression(max_iter=500)
    lr.fit(Reviews, Ratings)
    return lr

def ApplyGRU(Reviews, Ratings, EmbeddingLayer):
    model = Sequential()
    model.add(EmbeddingLayer)
    model.add(GRU(100))
    model.add(Dense(11, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 100
    epochs = 3
    model.fit(Reviews, Ratings, batch_size=batch_size, epochs=epochs)
    return model


def ApplyLSTM(Reviews, Ratings, EmbeddingLayer):
    model = Sequential()
    model.add(EmbeddingLayer)
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(11, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 100
    epochs = 3
    model.fit(Reviews, Ratings, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    return model


def ApplyMultinomialNB(Reviews, Ratings):
    mnb = MultinomialNB()
    mnb.fit(Reviews, Ratings)
    return mnb


def ApplySVM(Reviews, Ratings):
    svm = SGDClassifier(max_iter=1000)
    svm.fit(Reviews, Ratings)
    return svm
