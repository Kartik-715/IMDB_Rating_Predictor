from sklearn.metrics import accuracy_score

def findAccuracy(actual, prediction):
    accuracy = accuracy_score(actual, prediction)
    print("Accuracy of the model is", accuracy)