from sklearn.metrics import accuracy_score, f1_score

def compute(predictions, labels):
    """
    Calculate accuracy and F1 score.

    Parameters:
    - predictions: List of predicted labels
    - labels: List of true labels

    Returns:
    - accuracy: Accuracy score
    - f1: F1 score
    """
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Calculate F1 score
    f1 = f1_score(labels, predictions, average='weighted')

    return accuracy, f1
