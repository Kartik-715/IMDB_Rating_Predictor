from sklearn.model_selection import train_test_split

def split_data(reviews, rating):
    train_reviews, test_reviews, train_rating, test_rating = train_test_split(reviews, rating, test_size=0.25)
    print(train_reviews.shape)
    print(test_reviews.shape)
    return train_reviews, test_reviews, train_rating, test_rating
