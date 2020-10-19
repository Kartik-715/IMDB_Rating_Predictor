from sklearn.metrics import accuracy_score
def convertMatrixToArray(matrix):
    new_matrix = []
    for row in matrix:
        for i,val in enumerate(row):
            if(val == 1):
                new_matrix.append(i)
                break
    return new_matrix

def findAccuracy(actual, prediction, isMatrix):
    if isMatrix == True:
        actual = convertMatrixToArray(actual)
    correct = [1 if (abs(float(x) - float(y)) <= 1.0) else 0 for (x, y) in list(zip(actual, prediction))]
    accuracy = (sum(correct))/len(correct)
    accuracy = accuracy * 100
    print("Accuracy of the model is", accuracy)
