import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_PATH = 'data/'


def create_data_from_file(file_name: str, get_rules: bool = False):
    points = []
    with open(DATA_PATH + file_name, 'r') as text:
        for line in text.readlines():
            x, y, label = line.split()
            points.append(Point(float(x), float(y), int(label)))
    rules = None
    if get_rules:
        rules = create_lines(points)
    return points, rules


def split_data(points: list, split_size: float = 0.5):
    return train_test_split(points, test_size=split_size)


def create_lines(points):
    rules = []
    for i in range(len(points)):
        p1 = points[i]
        for j in range(i + 1, len(points)):
            p2=points[j]
            rules.append(Line(p1, p2,1))
            rules.append(Line(p1, p2,-1))
    return np.array(rules)


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


def plot_labels(X, predicted_labels):
    """
    Plot the predicted outputs labels
    """
    # green dots
    plt.scatter(x=X[predicted_labels == -1, 1], y=X[predicted_labels == -1, 0], alpha=0.9, c='green', marker='s', label=-1.0)

    # orange dots
    plt.scatter(x=X[predicted_labels == 1, 1], y=X[predicted_labels == 1, 0], alpha=0.9, c='orange', label=1.0)

    # location of the legend
    plt.title('Data Points')
    plt.legend(loc='upper left')
    plt.show()


""" Helping Data Objects """
class Point:
    def __init__(self, x: float, y: float, label: int, w: float = 0):
        self.x = x
        self.y = y
        self.label = label
        self.w = w


class Line:
    def __init__(self, p1: Point, p2: Point, direct: int):
        self.p1 = p1
        self.p2 = p2
        self.x = self.p1.x - self.p2.x
        self.y = self.p1.y - self.p2.y
        self.direct = direct
        self.w = 0

    def eval(self, p: Point):
        if self.x*(p.x-self.p2.x)-(p.x-self.p2.x)*self.y>=0:
            return self.direct
        else:
            return -self.direct