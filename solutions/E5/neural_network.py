import numpy as np
import h5py
import matplotlib.pyplot as plt
from copy import deepcopy


class NeuralNetwork:
    def __init__(self, sizes: list) -> None:
        self._num_layers = len(sizes) - 1
        self._sizes = sizes
        self._weights, self._bias = self._initialize_parameters()

    @staticmethod
    def load_data(dataset_file_handle: h5py.File):
        for i, key in enumerate(dataset_file_handle.keys()):
            match i:
                case 0:
                    list_of_classes = np.array(dataset_file_handle[key][:])
                case 1:
                    x_data = np.array(dataset_file_handle[key][:])
                case 2:
                    y_data = np.array(dataset_file_handle[key][:])

        return list_of_classes, x_data.reshape(x_data.shape[0], -1), y_data

    @staticmethod
    def sigmoid(Z: np.ndarray):
        return np.exp(Z) / (1 + np.exp(Z))

    @staticmethod
    def sigmoid_derivative(Z: np.ndarray):
        return NeuralNetwork.sigmoid(Z) * (1 - NeuralNetwork.sigmoid(Z))

    @staticmethod
    def relu(Z: np.ndarray):
        return (Z + np.abs(Z)) / 2
    
    @staticmethod
    def relu_derivarive(Z:np.ndarray):
        return 0 if Z < 0 else 1
    
    @staticmethod
    def mse(y_pred: np.ndarray, y_data: np.ndarray):
        return np.mean(np.power((y_data - y_pred), 2))

    @staticmethod
    def mse_derivative(y_pred: np.ndarray, y_data: np.ndarray):
        return 2 * (y_pred - y_data) / y_data.size

    def _initialize_parameters(self) -> list[np.ndarray]:
        sizes = self._sizes
        weights_list = list()
        bias_list = list()
        for i in range(self._num_layers):
            temp = np.sqrt(sizes[i])
            weights_list.append(np.random.uniform(-1 / temp, 1 / temp, (sizes[i + 1], sizes[i])))
            bias_list.append(np.random.uniform(-1, 1, sizes[i + 1]))
        return weights_list, bias_list

    def forward_propagation(self, X: np.ndarray):
        layers_output_list = list()
        A = X
        layers_output_list.append(A)
        for i in range(self._num_layers):
            Z = np.dot(self._weights[i], A) + self._bias[i]
            Y = self.sigmoid(Z)
            A = Y
            layers_output_list.append(A)
        return layers_output_list

    def backward_propagation(self, layers_output_list: list[np.ndarray], Y: np.ndarray):
        y_pred = layers_output_list[-1]
        dcdy = self.mse_derivative(y_pred, Y)
        dydz = self.sigmoid_derivative(y_pred)
        dw = [0] * self._num_layers
        db = [0] * self._num_layers

        for i in reversed(range(self._num_layers)):
            dcdw = np.outer((dcdy * dydz), layers_output_list[i].T)
            dcdb = dcdy
            dcdx = np.dot(self._weights[i].T, dcdy * dydz)
            dcdy = dcdx
            dydz = self.sigmoid_derivative(layers_output_list[i]).transpose()
            dw[i] = dcdw
            db[i] = dcdb
        return dw, db

    def update_parameters(self, learning_rate, grad_dw, grad_db):
        for i in range(self._num_layers):
            self._weights[i] -= learning_rate * grad_dw[i]
            self._bias[i] -= learning_rate * grad_db[i]

    @staticmethod
    def _sum_list(l1: list[np.ndarray], l2: list[np.ndarray]):
        result = list()
        for el1, el2 in zip(l1, l2):
            result.append(el1 + el2)
        return result

    @staticmethod
    def _scale_input(X: np.ndarray):
        return X / 255

    def train_model(self, train_data_file_handle: h5py.File, iterations: int, learning_rate: float):
        classes, train_data_x, train_data_y = self.load_data(train_data_file_handle)
        for i in range(iterations):
            print(i)
            grad_dw_sum = None
            grad_db_sum = None

            for X, Y in zip(train_data_x, train_data_y):
                X = self._scale_input(X)
                A = self.forward_propagation(X)
                dw, db = self.backward_propagation(A, Y)
                if grad_dw_sum and grad_db_sum:
                    grad_dw_sum=self._sum_list(grad_dw_sum, dw)
                    grad_db_sum=self._sum_list(grad_db_sum, db)
                else:
                    grad_dw_sum = dw
                    grad_db_sum = db

            grad_dw = [el / train_data_y.size for el in grad_dw_sum]
            grad_db = [el / train_data_y.size for el in grad_db_sum]
            self.update_parameters(learning_rate, grad_dw, grad_db)

    def predict(self,X):
        threshold = 0.5
        output = self.forward_propagation(X)
        return( 1 if output[-1]>threshold else 0), output[-1]

    def evaluate_model(self, test_data_file_handle:h5py.File):
        classes, test_data_x, test_data_y = self.load_data(test_data_file_handle)
        correct_predictions = 0
        cost = 0
        for X, Y in zip(test_data_x,test_data_y):
            # print(Y)
            y_pred, out = self.predict(self._scale_input(X))
            correct_predictions+=int(y_pred==Y)
            cost+=self.mse(out,Y)
        return cost/test_data_y.size, correct_predictions

        

with open('output1.txt', 'w') as file:

    iterations = [50,100,500,1000,5000,10000]
    neurons = [3,4,5,6,7,9,15,25]
    learning_rate = [0.01,0.1,0.3,0.5,0.75]
    # print("Checking iterations")
    # for iter in  iterations:
    #     i = iter
    #     lr = 0.7
    #     neur = 7
    #     with h5py.File("data/train_catvnoncat.h5") as train_file:
    #         network = NeuralNetwork([64 * 64 * 3, neur, 1])
    #         network.train_model(train_file,i,lr)
    #         with h5py.File("data/test_catvnoncat.h5") as test_flie:
    #             mean_cost, accuracy = network.evaluate_model(test_flie)
    #             file.write(f"{i},{neur},{lr},{mean_cost},{accuracy}\n")

    print("Checking neurons")

    for n in  neurons:
        i = 2000
        lr = 0.7
        neur = n
        with h5py.File("data/train_catvnoncat.h5") as train_file:
            network = NeuralNetwork([64 * 64 * 3, neur, 1])
            network.train_model(train_file,i,lr)
            with h5py.File("data/test_catvnoncat.h5") as test_flie:
                mean_cost, accuracy = network.evaluate_model(test_flie)
                file.write(f"{i},{neur},{lr},{mean_cost},{accuracy}\n")

    print("Checking learning rate")

    for rate in  learning_rate:
        i = 2000
        lr = rate
        neur = 7
        with h5py.File("data/train_catvnoncat.h5") as train_file:
            network = NeuralNetwork([64 * 64 * 3, neur, 1])
            network.train_model(train_file,i,lr)
            with h5py.File("data/test_catvnoncat.h5") as test_flie:
                mean_cost, accuracy = network.evaluate_model(test_flie)
                file.write(f"{i},{neur},{lr},{mean_cost},{accuracy}\n")


