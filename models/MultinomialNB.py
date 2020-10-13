from sklearn.naive_bayes import MultinomialNB

def ApplyMultinomialNB(Reviews, Ratings):
    mnb = MultinomialNB()
    mnb.fit(Reviews, Ratings)
    return mnb
