from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score

''' Pass Data and the function will return the trained model '''
def ApplyLogisticRegression(Reviews, Ratings):
    lr = LogisticRegression(max_iter=500)
    xx = lr.fit(Reviews, Ratings)
    print(xx)
    return lr


def predictLR(lr_model: LogisticRegression, test_reviews):
    return lr_model.predict(test_reviews)


''' Pass Data and the function will return the trained model '''
def ApplySVM(Reviews, Ratings):
    svm = SGDClassifier(max_iter=500)
    xx = svm.fit(Reviews, Ratings)
    print(xx)
    return svm


def predictSVM(svm_model: SGDClassifier, test_reviews):
    return svm_model.predict(test_reviews)


def findAccuracy(actual, prediction):
    accuracy = accuracy_score(actual, prediction)
    print("Accuracy of the model is", accuracy)

