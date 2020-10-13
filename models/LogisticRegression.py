from sklearn.linear_model import LogisticRegression

''' Pass Data and the function will return the trained model '''
def ApplyLogisticRegression(Reviews, Ratings):
    lr = LogisticRegression(max_iter=500)
    lr.fit(Reviews, Ratings)
    return lr

