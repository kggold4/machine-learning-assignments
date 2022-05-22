import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'data/'


def create_data_from_file(file_name: str):
    data = []
    label = []
    with open(DATA_PATH + file_name, 'r') as text:
        for line in text.readlines():
            _x, _y, _label = line.split()
            data.append((float(_x),float(_y)))
            label.append(int(_label))
    label = np.array(label)
    return np.array(data), np.array(label)


def plot_labels(X, predicted_labels):
    """
    Plot the predicted outputs labels
    """
    # green dots
    plt.scatter(x=X[predicted_labels == -1, 1], y=X[predicted_labels == -1, 0], alpha=0.9, c='green', marker='s', label=-1.0)

    # orange dots
    plt.scatter(x=X[predicted_labels == 1, 1], y=X[predicted_labels == 1, 0], alpha=0.9, c='orange', label=1.0)

    # location of the legend
    plt.rcParams["figure.figsize"] = (7, 7)
    plt.legend(loc='upper left')
    plt.show()


def mse(real, output) -> float:
    """
    Calculate the mean square error of predicted label data and the real label data
    """
    return ((np.array(real) - np.array(output))**2).mean()


def accuracy(predicted, test_label) -> float:
    """
    Given a predicted label and the test label return the accuracy
    """
    count = 0
    for i, j in zip(predicted, test_label):
        if i == j:
            count += 1
    return count / len(predicted)
