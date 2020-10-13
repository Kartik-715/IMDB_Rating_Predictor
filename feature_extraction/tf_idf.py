from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(train_reviews, test_reviews):
    tv=TfidfVectorizer(use_idf=True,ngram_range=(1,3))
    tv_train_reviews=tv.fit_transform(train_reviews)
    tv_test_reviews=tv.transform(test_reviews)
    return tv_train_reviews, tv_test_reviews
