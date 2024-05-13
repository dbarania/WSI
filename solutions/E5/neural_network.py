import numpy as np
import h5py
import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self,sizes:list) -> None:
        self._num_layers = len(sizes)-1
        self._sizes = sizes
        self._weights = self._initialize_weights()


    @staticmethod
    def load_data(dataset_file_handle:h5py.File):
        for i,key in enumerate(dataset_file_handle.keys()):
            match i:
                case 0:
                    list_of_classes = np.array(dataset_file_handle[key])
                case 1:
                    x_data = np.array(dataset_file_handle[key])
                case 2:
                    y_data = np.array(dataset_file_handle[key])
                            
        return list_of_classes,x_data,y_data
    

    @staticmethod
    def sigmoid(Z:np.ndarray):
        return np.exp(Z)/(1+np.exp(Z))
    
    @staticmethod
    def relu(Z:np.ndarray):
        return (Z+np.abs(Z))/2
    
    @staticmethod
    def softmax(Z:np.ndarray):
        return np.exp(Z)/np.sum(np.exp(Z))

    def _initialize_weights(self)->list[np.ndarray]:
        sizes = self._sizes
        weights_list = list()
        for i in range(self._num_layers):
            if i == self._num_layers-1:
                weights_list.append(np.zeros((sizes[i+1],sizes[i]+1)))
            else:
                temp = np.sqrt(sizes[i])
                weights_list.append(np.random.uniform(-1/temp,1/temp,(sizes[i+1],sizes[i]+1)))
        return weights_list
    
    def feed_forward(self,X:np.ndarray):
        pass
