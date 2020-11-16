import pandas as pd
import numpy as np
import random
from os import listdir
from os.path import isfile, join
from preprocessing.clean import preprocess_data
from feature_extraction import split
from feature_extraction import bag_of_words
from feature_extraction import tf_idf
from feature_extraction import glove
from feature_extraction import word_2_vec
from models.model import *
from Metrics.helpers import findAccuracy
from tensorflow.keras.initializers import Constant
from tensorflow.python.keras.layers import Embedding

CORPUS_FOLDER = './corpus'
CLEANED_CORPUS_FOLDER = './cleaned_corpus'
SHORT_NUM_MOVIES=3

def read_raw_data():
    FILE_NAMES = [f for f in listdir(CORPUS_FOLDER) if
                  isfile(join(CORPUS_FOLDER, f))]  # Get List of all files in the CORPUS_FOLDER
    FILE_NAMES = random.sample(FILE_NAMES, SHORT_NUM_MOVIES)

    for filename in FILE_NAMES:
        imdb_data = pd.read_csv(CORPUS_FOLDER + '/' + filename, sep='\t', header=0)
        print(imdb_data.shape)
        imdb_data = preprocess_data(imdb_data)
        imdb_data.to_csv(CLEANED_CORPUS_FOLDER + '/' + filename, sep='\t', index=False)


def read_clean_data():
    FILE_NAMES = [f for f in listdir(CLEANED_CORPUS_FOLDER) if
                  isfile(join(CLEANED_CORPUS_FOLDER, f))]  # Get List of all files in the CLEANED_CORPUS_FOLDER

    reviews = np.empty([0, 1])
    rating = np.empty([0, 1])
    for filename in FILE_NAMES:
        imdb_data = pd.read_csv(CLEANED_CORPUS_FOLDER + '/' + filename, sep='\t', header=0)
        reviews = np.append(reviews, imdb_data.Text.apply(lambda x: np.str_(x)))
        rating = np.append(rating, imdb_data.Rating.apply(lambda x: np.str_(x)))
    return reviews, rating

def main():
    read_raw_data()
    reviews, rating = read_clean_data()

    # train_reviews, test_reviews, train_rating, test_rating = split.split_data(reviews, rating)
    # train_reviews, test_reviews = bag_of_words.bag_of_words(train_reviews, test_reviews)
    # train_reviews, test_reviews = tf_idf.tf_idf(train_reviews, test_reviews)
    # train_reviews, test_reviews, train_rating, test_rating, embedding_matrix, num_words, embeddingDim, maxLength = word_2_vec.word_2_vec(reviews, rating)
    # print(train_reviews.shape)
    # print(test_reviews.shape)
    # train_reviews, test_reviews, train_rating, test_rating, embedding_matrix, num_words, embeddingDim, maxLength = glove.glove(reviews, rating)
    # print(train_reviews.shape)
    # print(test_reviews.shape)
    # print(train_rating.shape)
    # print(test_rating.shape)
    # model = ApplyLogisticRegression(train_reviews, train_rating)
    # model = ApplySVM(train_reviews, train_rating)
    # model = ApplyMultinomialNB(train_reviews, train_rating)
    # model = ApplyLSTM(train_reviews, train_rating, Embedding(num_words, embeddingDim, embeddings_initializer = Constant(embedding_matrix), input_length = maxLength, trainable = False))
    # model = ApplyGRU(train_reviews, train_rating, Embedding(num_words, embeddingDim, embeddings_initializer = Constant(embedding_matrix), input_length = maxLength, trainable = False))
    predicted_ratings = model.predict_classes(test_reviews)
    # predicted_ratings = model.predict(test_reviews)
    # findAccuracy(test_rating, predicted_ratings, True)


if __name__ == '__main__':
    main()
