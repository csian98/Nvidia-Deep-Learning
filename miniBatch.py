#!/opt/homebrew/Caskroom/miniforge/base/envs/sian/bin/python3

import numpy as np

class MiniBatch:
    def __init__(self, input_sz, output_sz, hidden_sz, BATCH_SIZE=32):
        self.BATCH_SIZE=BATCH_SIZE
        self.hidden_layer_w=self.layer_w(hidden_sz, input_sz)
        self.hidden_layer_y=np.zeros((hidden_sz, self.BATCH_SIZE))
        self.hidden_layer_error=np.zeros((hidden_sz, self.BATCH_SIZE))

        self.output_layer_w=self.layer_w(output_sz, hidden_sz)
        self.output_layer_y=np.zeros((output_sz, self.BATCH_SIZE))
        self.output_layer_error=np.zeros((output_sz, self.BATCH_SIZE))

    def forward(self, x):
        hidden_layer_z=np.matmul(self.hidden_layer_w, x)
        self.hidden_layer_y=np.tanh(hidden_layer_z)
        hidden_output_array=np.concatenate(
            (np.ones((1, self.BATCHSIZE)), self.hidden_layer_y)
        )

        output_layer_z=np.matmul(self.output_layer_w, hidden_output_array)
        self.output_layer_y=1.0/(1.0+np.exp(-output_layer_z))
        
    def backward(self, y_true):
        error_prime=-(y_true-self.output_layer_y)
        output_logistic_prime=self.output_layer_y*(1.0-self.output_layer_y)
        self.output_layer_error=error_prime*output_logistic_prime

        hidden_tanh_prime=1.0-self.hidden_layer_y**2
        hidden_layer_weighted_error=np.matmul(
            np.matrix.transpose(
                self.output_layer_w[:, 1:]
            ), self.output_layer_error
        )
        self.hidden_layer_error=hidden_tanh_prime*hidden_layer_weighted_error
        
    def adjust_weights(self, x, LEARNING_RATE=0.01):
        delta_matrix=np.zeros((len(self.hidden_layer_error[:, 0]), len(x[:, 0])))
        for i in range(self.BATCH_SIZE):
            delta_matrix+=np.outer(self.hidden_layer_error[:, i], x[:, i])*LEARNING_RATE

        delta_matrix/=self.BATCH_SIZE
        self.hidden_layer_w-=delta_matrix

        hidden_output_array=np.concatenate(
            (np.ones((1, self.BATCH_SIZE)), self.hidden_layer_y)
        )
        delta_matrix=np.zeros((len(self.output_layer_error[:, 0]),
                               len(hidden_output_array[:, 0])))
        for i in range(self.BATCH_SIZE):
            delta_matrix+=np.outer(self.output_layer_error[:, i], hidden_output_array[:, i])*LEARNING_RATE
        
        delta_matrix/=self.BATCH_SIZE
        self.output_layer_w-=delta_matrix
    
    def layer_w(self, neuron_count, input_count):
        weights=np.zeros((neuron_count, input_count+1))
        for i in range(neuron_count):
            for j in range(1, input_count+1):
                weights[i][j]=np.random.uniform(-0.1, 0.1)

        return weights

if __name__ == "__main__":
    dnn=MiniBatch(784, 10, 25)

    BATCH_SIZE=32
    LEARNING_RATE=0.01
    EPOCH=10

    index_list=list(range(int(len(x_train)/BATCH_SIZE)))
