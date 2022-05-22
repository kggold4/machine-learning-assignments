import numpy as np
import matplotlib_inline as plt

def create_data_from_file(file_name: str):
    data = []
    label = []
    with open(file_name, 'r') as text:
        for line in text.readlines():
            _x, _y, _label = line.split()
            data.append((float(_x),float(_y)))
            label.append(int(_label))
    label = np.array(label)
    return np.array(data), np.array(label)

def plot_labels(x, model):
    """
    Plot the outputs of each neuron in the given layer
    """
    neuron_number = 0

    # for neuron in layer_output:
    y_hat=model.predict(x)
    # green dots
    plt.scatter(x=x[y_hat == -1, 1], y=x[y_hat == -1, 0], alpha=0.9,
                    c='green', marker='s', label=-1.0)

    # orange dots
    plt.scatter(x=x[y_hat == 1, 1], y=x[y_hat == 1, 0], alpha=0.9,
                    c='orange', label=1.0)

    # location of the legend
    plt.legend(loc='upper left')
    plt.show()
