from sklearn.linear_model import SGDClassifier

''' Pass Data and the function will return the trained model '''
def ApplySVM(Reviews, Ratings):
    svm = SGDClassifier(max_iter=1000)
    svm.fit(Reviews, Ratings)
    return svm