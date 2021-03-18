import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import copy

class FFNN(nn.Module):
    """
    simple FFNN with dropout regularization
    """
    def __init__(self, dim_in, hidden, dim_out, CUDA=False, SEED=None, output_limit=None, dropout=0.0, hidden_activation="tanh"):

        super(FFNN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.output_limit=output_limit
        self.hidden = hidden
        self.hidden_activation = nn.ReLU() if hidden_activation=="relu" else nn.Tanh()
        if not SEED == None:
            torch.manual_seed(SEED)
            if CUDA:
                torch.cuda.manual_seed(SEED)

        self.Layers = nn.ModuleList()
        self.Layers.append(nn.Linear(dim_in, hidden[0]))
        for i in range(0,len(hidden)-1):
            self.Layers.append(nn.Linear(hidden[i], hidden[i+1]))        
        self.fcout = nn.Linear(hidden[-1], dim_out)

        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = []
        outputs.append(self.hidden_activation(self.Layers[0](x)))
        
        if self.training:
            for i in range(1, len(self.hidden)):
                outputs.append(self.hidden_activation(self.dropout(self.Layers[i](outputs[i-1]))))
        else:
            for i in range(1, len(self.hidden)):
                outputs.append(self.hidden_activation(self.Layers[i](outputs[i-1])))
            
        if self.output_limit is None:
            return self.fcout(outputs[-1])
        else:
            return  self.Tanh(self.fcout(outputs[-1])) * self.output_limit

    # Call this when not training
    # Other wise repeatedly calling forward will consume memory by building the 
    # computation graph longer and longer
    def predict(self, x):
        outputs = []
        outputs.append(self.hidden_activation(self.Layers[0](x)))
        
        if self.training:
            for i in range(1, len(self.hidden)):
                outputs.append(self.hidden_activation(self.dropout(self.Layers[i](outputs[i-1]))))
        else:
            for i in range(1, len(self.hidden)):
                outputs.append(self.hidden_activation(self.Layers[i](outputs[i-1])))
            
        if self.output_limit is None:
            return self.fcout(outputs[-1]).detach()
        else:
            return  (self.Tanh(self.fcout(outputs[-1])) * self.output_limit).detach()
    
    def loss_function(self, y, y_pred):
        MSE = (y - y_pred).pow(2).sum()
        return MSE
    

class FFNN_Model():
    def __init__(self, dim_in, hidden, dim_out, CUDA=False, SEED=None, output_limit=None, dropout=0.0, hidden_activation="tanh"):
       
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = hidden
        self.CUDA = CUDA
        self.output_limit=output_limit
        self.model = FFNN (dim_in=dim_in, hidden=hidden, dim_out=dim_out, CUDA=CUDA, SEED=SEED, output_limit=output_limit, dropout=dropout, hidden_activation=hidden_activation)
        self.data_mean_input = None
        self.data_mean_output = None
        self.data_std_input = None
        self.data_std_output = None
        
        if CUDA:
            self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
    def train(self, epochs, training_inputs, training_targets, batch_size=None, logInterval=100):
        """
        epochs           : Gradient updates per mini-batch
        training_data    : Data for which the distribution to be learnt by the CVAE
        descriptors_data : Descriptor input for the training data points. It's the condition input for the network. 
        """
        self.data_mean_input = torch.FloatTensor(np.mean(training_inputs, axis=0)).cuda() if self.CUDA else torch.FloatTensor(np.mean(training_inputs, axis=0))
        self.data_std_input = torch.FloatTensor(np.std(training_inputs, axis=0)).cuda() + 1e-10 if self.CUDA else torch.FloatTensor(np.std(training_inputs, axis=0)) + 1e-10
        self.data_mean_output = torch.FloatTensor(np.mean(training_targets, axis=0)).cuda() if self.CUDA else torch.FloatTensor(np.mean(training_targets, axis=0))
        self.data_std_output = torch.FloatTensor(np.std(training_targets, axis=0)).cuda() + 1e-10 if self.CUDA else torch.FloatTensor(np.std(training_targets, axis=0)) + 1e-10

        training_inputs_tensor = torch.FloatTensor(training_inputs).cuda() if self.CUDA else torch.FloatTensor(training_inputs)
        training_targets_tensor = torch.FloatTensor(training_targets).cuda() if self.CUDA else torch.FloatTensor(training_targets)
        
        training_inputs_tensor = (training_inputs_tensor - self.data_mean_input) / self.data_std_input
        training_targets_tensor = (training_targets_tensor - self.data_mean_output) / self.data_std_output
        
        if self.CUDA:
            training_inputs_tensor = training_inputs_tensor.cuda()
        
        if batch_size is None:
            batch_size = len(training_inputs)
        
        if batch_size > len(training_inputs):
            batch_size = len(training_inputs)

        mini_batches = int(np.ceil(float(len(training_inputs))/float(batch_size)))
        loss_vals = []
        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(training_inputs_tensor.size()[0])
            for i in range(mini_batches):
                x = training_inputs_tensor[permutation[i*batch_size : i*batch_size + batch_size]]
                y = training_targets_tensor[permutation[i*batch_size : i*batch_size + batch_size]]
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.model.loss_function(y, y_pred)
                loss.backward()
                self.optimizer.step()
                loss_vals.append(loss.item())
            if logInterval is not None:
                if epoch%logInterval == 0:
                    print("Loss: ", np.mean(loss_vals)/float(batch_size))
                    loss_vals = []    
        self.model.train(mode=False)
    
    def predict(self, d_in): #TODO: this is not efficient due to the conversions from numpy to tensor
        """
       d_in: 2d numpy arrays
        """
        x = (torch.FloatTensor(d_in).cuda() - self.data_mean_input) / self.data_std_input if self.CUDA else (torch.FloatTensor(d_in) - self.data_mean_input) / self.data_std_input

        return ((self.model.forward(x) * self.data_std_output) + self.data_mean_output).cpu().detach().numpy()
    
    def predict_tensor(self, d_in):
        """
       d_in: 2d tensor. Must be casted properly to cpu or cuda.
        """
        return self.model.predict((d_in - self.data_mean_input) / self.data_std_input) * self.data_std_output + self.data_mean_output
    

class FFNN_Ensemble_Model():
    def __init__(self, dim_in, hidden, dim_out, n_ensembles, CUDA=False, SEED=None, output_limit=None, dropout=0.0, hidden_activation="tanh"):       
        self.n_ensembles = n_ensembles
        self.models = []
        self.CUDA = CUDA
        for i in range(self.n_ensembles):
            self.models.append(FFNN_Model(dim_in=dim_in, hidden=hidden, dim_out=dim_out, CUDA=CUDA, SEED=SEED, output_limit=output_limit, dropout=dropout, hidden_activation=hidden_activation))
        
    def train(self, epochs, training_inputs, training_targets, sampling_size, batch_size=None, logInterval=100):
        for i in range(self.n_ensembles):
            sampled_indices = range(len(training_inputs))
            if self.n_ensembles > 1 and sampling_size > 0 :
                print("Data randomly sampled with replacement: ", sampling_size, " times")
                sampled_indices =  np.random.randint(0, len(training_inputs), size=sampling_size)
            self.models[i].train(epochs=epochs, training_inputs=training_inputs[sampled_indices], training_targets=training_targets[sampled_indices], batch_size=batch_size, logInterval=logInterval)
    
    def forward(self, d_in):
        """
       d_in: 2d numpy arrays
        """
        y_pred = []
        for i in range(self.n_ensembles):
            y_pred.append(self.models[i].forward(d_in))
        
        return np.mean(y_pred, axis=0), np.var(y_pred, axis=0)
    
    def predict(self, d_in):
        """
       d_in: 2d numpy arrays
        """
        return self.forward(d_in)

    def pred_mu(self, d_in):
        y_pred = []
        for i in range(self.n_ensembles):
            y_pred.append(self.models[i].forward(d_in))
        
        return np.mean(y_pred, axis=0)
    
    def pred_var(self, d_in):
        y_pred = []
        for i in range(self.n_ensembles):
            y_pred.append(self.models[i].forward(d_in))
        
        return np.var(y_pred, axis=0)
    
    def get_models(self):
        return copy.deepcopy(self.models)


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import time

    x = np.linspace(-6, 6, 100).reshape(100,-1)
    y = np.sin(x)
    print(x)

    network = FFNN_Model(dim_in=1, hidden=[10, 10, 10], dim_out=1, CUDA=False, SEED=None, output_limit=None, dropout=0.0)
    network.train(epochs=2000, training_inputs=x, training_targets=y, batch_size=32, logInterval=100)
    y_pred = network.predict(y)
    plt.plot(x.flatten(), y_pred.flatten(), '-b')
    plt.plot(x.flatten(), y.flatten(), '--b')
    plt.show()
