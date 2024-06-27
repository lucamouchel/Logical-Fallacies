from sklearn.metrics import accuracy_score, f1_score

def compute(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return accuracy, f1
