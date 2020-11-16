from keras.layers import Bidirectional, GlobalMaxPool1D, GRU
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from keras.layers import concatenate
from keras import Model


''' Pass Data and the functions will return the trained model '''

def ApplyLogisticRegression(Reviews, Ratings):
    lr = LogisticRegression(penalty='l2',dual=True,tol=0.0001,solver='newton-cg',C=0.9)
    lr.fit(Reviews, Ratings)
    return lr

def ApplyCNN(Reviews, Ratings, EmbeddingLayer, maxLength):
    sequence_input = Input(shape=(maxLength,), dtype='int32')
    embedded_sequences = EmbeddingLayer(sequence_input)
    convs = []
    filter_sizes = [2,3,4,5,6]
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200,
                        kernel_size=filter_size,
                        activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)
    l_merge = concatenate(convs, axis=1)
    x = Dropout(0.1)(l_merge)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(11, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    num_epochs = 3
    batch_size = 32
    hist = model.fit(Reviews, Ratings, epochs=num_epochs, validation_split=0.1, shuffle=True, batch_size=batch_size)
    return model

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
    mnb = MultinomialNB(alpha=0.8)
    mnb.fit(Reviews, Ratings)
    return mnb


def ApplySVM(Reviews, Ratings):
    svm = SGDClassifier(max_iter=1000)
    svm.fit(Reviews, Ratings)
    return svm
