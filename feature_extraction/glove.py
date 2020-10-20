import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from feature_extraction import split


# Using Google pre-trained glove embeddings
def glove(reviews, rating):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews)
    # list of reviews (Each review -> sequence of word index)
    sequences = tokenizer.texts_to_sequences(reviews)
    word_index = tokenizer.word_index
    maxLength = 0
    for review in sequences:
        if maxLength < len(review):
            maxLength = len(review)

    embeddings = {}
    fp = open("./embeddings/glove.twitter.27B.200d.txt")
    for line in fp:
        val = line.split()
        word = val[0]
        vector_coefs = np.asarray(val[1:])
        if word in word_index.keys():
            embeddings[word] = vector_coefs
    fp.close()

    review_pad = pad_sequences(sequences, maxlen = maxLength)
    rating = np.asarray(rating)

    embeddingDim = 200
    num_words = len(word_index)+1
    embedding_matrix  = np.zeros((num_words, embeddingDim))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    rating_checklist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    rating_matrix = np.zeros((len(rating), 11))
    for i in range(len(rating)):
        if rating[i] in rating_checklist:
            rating_matrix[i][int(rating[i])] = 1

    train_reviews, test_reviews, train_rating, test_rating = split.split_data(review_pad, rating_matrix)
    return train_reviews, test_reviews, train_rating, test_rating, embedding_matrix, num_words, embeddingDim, maxLength
