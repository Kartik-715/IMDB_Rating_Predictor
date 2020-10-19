from sklearn.metrics import accuracy_score

def findAccuracy(actual, prediction):
    correct = [1 if (abs(float(x) - float(y)) <= 1.0) else 0 for (x, y) in list(zip(actual, prediction))]
    accuracy = (sum(correct))/len(correct)
    accuracy = accuracy * 100
    print("Accuracy of the model is", accuracy)