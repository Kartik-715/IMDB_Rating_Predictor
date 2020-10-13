from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(train_reviews, test_reviews):
    cv=CountVectorizer(binary=False,ngram_range=(1,3))
    cv_train_reviews=cv.fit_transform(train_reviews)
    cv_test_reviews=cv.transform(test_reviews)
    return cv_train_reviews, cv_test_reviews
