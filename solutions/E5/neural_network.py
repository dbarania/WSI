import numpy as np
import h5py
import matplotlib.pyplot as plt
from copy import deepcopy
class NeuralNetwork:
    def __init__(self,sizes:list) -> None:
        self._num_layers = len(sizes)-1
        self._sizes = sizes
        self._weights, self._bias  = self._initialize_weights()


    @staticmethod
    def load_data(dataset_file_handle:h5py.File):
        for i,key in enumerate(dataset_file_handle.keys()):
            match i:
                case 0:
                    list_of_classes = np.array(dataset_file_handle[key][:])
                case 1:
                    x_data = np.array(dataset_file_handle[key][:])
                case 2:
                    y_data = np.array(dataset_file_handle[key][:])
                            
        return list_of_classes,x_data,y_data
    

    @staticmethod
    def sigmoid(Z:np.ndarray):
        return np.exp(Z)/(1+np.exp(Z))
    
    @staticmethod
    def sigmoid_derivative(Z:np.ndarray):
        return NeuralNetwork.sigmoid(Z)*(1-NeuralNetwork.sigmoid(Z))

    @staticmethod
    def relu(Z:np.ndarray):
        return (Z+np.abs(Z))/2
    
    @staticmethod
    def mse(y_pred:np.ndarray,y_data:np.ndarray):
        return np.mean(np.power((y_data-y_pred),2))    
    
    @staticmethod
    def mse_derivative(y_pred:np.ndarray,y_data:np.ndarray):
        return 2*(y_pred-y_data)/y_data.size

    def _initialize_weights(self)->list[np.ndarray]:
        sizes = self._sizes
        weights_list = list()
        bias_list = list()
        for i in range(self._num_layers):
            temp = np.sqrt(sizes[i])
            weights_list.append(np.random.uniform(-1/temp,1/temp,(sizes[i+1],sizes[i])))
            bias_list.append(np.random.uniform(-1,1,sizes[i+1]))
        return weights_list, bias_list
  
    def forward_propagation(self, X:np.ndarray):
        layers_output_list = list()
        A = X
        layers_output_list.append(A)
        for i in range(self._num_layers):
            # Z = self._weights[i].dot(A)
            # Y = self.sigmoid(Z)
            # A = np.append(Y,1)
            Z = np.dot(self._weights[i],A)+self._bias[i]
            Y = self.sigmoid(Z)
            A = Y
            layers_output_list.append(A)
        # layers_output_list[-1] = np.delete(layers_output_list[-1],-1,0)
        return layers_output_list
    
    def backward_propagation(self, layers_output_list:list[np.ndarray], Y:np.ndarray):
        y_pred = layers_output_list[-1]
        dcdy = self.mse_derivative(y_pred,Y)
        dydz = self.sigmoid_derivative(y_pred)
        dw = [0]*self._num_layers
        db = [0]*self._num_layers


        for i in reversed(range(self._num_layers)):
            # dCdw = np.multiply(dCdY,dydz_old).dot(layers_output_list[i])
                # dCdw = np.multiply(dCdY,dydz_old).dot(layers_output_list[i]) if np.multiply(dCdY,dydz_old).shape!=np.zeros((1)).shape \
                #     else np.multiply(np.multiply(dCdY,dydz_old),(layers_output_list[i]))
                # dCdY = self._weights[i].transpose().dot(np.multiply(dCdY,dydz_old))
            # dCdw = np.outer(np.multiply(dCdY.transpose()[0],dydz_old),layers_output_list[i])
            # dCdX = np.multiply(self._weights[i], np.multiply(dCdY.transpose(),dydz_old))
            # dCdY = np.delete(dCdX,-1)
            dcdw = np.outer((dcdy*dydz),layers_output_list[i].T)
            dcdb = dcdy
            dcdx = np.dot(self._weights[i].T,dcdy*dydz)
            dcdy = dcdx
            dydz = self.sigmoid_derivative(layers_output_list[i]).transpose()
            dw[i] = dcdw
            db[i] = dcdb
        return dw, db
       
    def update_parameters(self,learing_late,grad_dw,grad_db):
        for i in range(self._num_layers):
            self._weights[i]-=learing_late*grad_dw[i]
            self._bias[i]-=learing_late*grad_db[i]


    def train_model(self, train_data_file_handle:h5py.File):
        classes, train_data_x,train_data_y = self.load_data(train_data_file_handle)
        print(classes)
        
with h5py.File("data/train_catvnoncat.h5") as train_file:
    network = NeuralNetwork([64*64*3,10,1])
    network.train_model(train_file)