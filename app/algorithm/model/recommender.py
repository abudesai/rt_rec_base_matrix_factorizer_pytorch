import sys

import numpy as np, pandas as pd
import os
from sklearn.utils import shuffle
import joblib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import GRU, LSTM, ReLU, Linear, Embedding, Module, CrossEntropyLoss

from torch.utils.data import Dataset, DataLoader

MODEL_NAME = "MatrixFactorizer_using_GradDescent_PyTorch"


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"


def get_loss(model, device, data_loader, loss_function):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for data in data_loader:
            input, label = data[0].to(device), data[1].to(device)
            output = model(input.view(input.shape[0], -1))
            loss = loss_function(output, label)
            loss_total += loss.item()
    return loss_total / len(data_loader)



class Net(Module):
    def __init__(self, N, M, K):
        super().__init__()
        self.u_embedding = nn.Embedding(N, K)  # shape => (N, 1, K)
        self.m_embedding = nn.Embedding(M, K)  # shape => (N, 1, K)
        self.u_bias = nn.Embedding(N, 1)  # shape => (N, 1, 1)
        self.m_bias = nn.Embedding(M, 1)  # shape => (N, 1, 1)

    def forward(self, x):     
        u, m = torch.split(x, [1,1], dim=1)           
        t = torch.squeeze(torch.sum( self.u_embedding(u) * self.m_embedding(m), dim=2))
        t = t + torch.squeeze(self.u_bias(u)) + torch.squeeze(self.m_bias(m)) 
        return t
    
    def get_num_parameters(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):                
                nn = nn*s
            pp += nn
        return pp    



class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self,index):
        # Get one item from the dataset
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)
    
    

class Recommender():
    def __init__(self, N, M, K, lr=0.1, **kwargs):
        '''
        N = num users
        M = num items
        K = embedding dimension size
        
        '''
        self.N = N
        self.M = M
        self.K = K
        self.lr = lr

        self.model = Net(
            N=self.N,
            M=self.M,
            K=self.K
        )
        self.model.to(device)
        
        # print(self.model.get_num_parameters())
        # sys.exit()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.print_period = 5


    # def __call__(self, X):
    #     X = torch.LongTensor(X)
    #     preds = self.model(X).detach().cpu().numpy()
    #     return preds


    def fit(self, train_X, train_y, valid_X=None, valid_y=None, epochs=100, batch_size=32, verbose=0):        
        
        train_X, train_y = torch.LongTensor(train_X), torch.LongTensor(train_y)
        train_dataset = CustomDataset(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=int(batch_size), shuffle=True)        
        
        if valid_X is not None and valid_y is not None:
            valid_X, valid_y = torch.LongTensor(valid_X), torch.LongTensor(valid_y)   
            valid_dataset = CustomDataset(valid_X, valid_y)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=int(batch_size),  shuffle=True)
        else:
            valid_loader = None

        losses = self._run_training(train_loader, valid_loader, epochs,
                           use_early_stopping=True, patience=5,
                           verbose=verbose)
        return losses
    
    
    def _run_training(self, train_loader, valid_loader, epochs,
                      use_early_stopping=True, patience=10, verbose=1):
        best_loss = 1e7
        losses = []
        min_epochs = 1
        for epoch in range(epochs):
            self.model.train()
            for s, data in enumerate(train_loader, 0):
                inputs,  labels = data[0].to(device), data[1].to(device)
                # print(inputs); sys.exit()
                # Feed Forward
                output = self.model(inputs)
                # Loss Calculation
                loss = self.criterion(output, labels)
                # Clear the gradient buffer (we don't want to accumulate gradients)
                self.optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Weight Update: w <-- w - lr * gradient
                self.optimizer.step()
                
                current_loss = loss.item()    
                if verbose:
                    print(f"Epoch: {s+1}/{epochs}, step: {s}, Training Loss: {current_loss}. ")
                
            current_loss = loss.item()            
            
            if use_early_stopping:
                # Early stopping
                if valid_loader is not None:
                    current_loss = get_loss(self.model, device, valid_loader, self.criterion)
                losses.append({"epoch": epoch, "loss": current_loss})
                if current_loss < best_loss:
                    trigger_times = 0
                    best_loss = current_loss
                else:
                    trigger_times += 1
                    if trigger_times >= patience and epoch >= min_epochs:
                        if verbose == 1: print('Early stopping!')
                        return losses
                    
                
                
            else:
                losses.append({"epoch": epoch, "loss": current_loss})
            # Show progress
            if verbose == 1:
                if epoch % self.print_period == 0 or epoch == epochs-1:
                    print(f'Epoch: {epoch+1}/{epochs}, loss: {np.round(loss.item(), 5)}')
        return losses   


    def predict(self, X):
        X = torch.LongTensor(X).to(device)
        preds = self.model(X).detach().cpu().numpy()
        return preds

    def summary(self):
        print(self.model)

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return the loss and metrics"""
        return self.model.evaluate(
            x=[x_test[:, 0], x_test[:, 1]],
            y=y_test,
            verbose=0)

    def save(self, model_path):
        model_params = {
            "N": self.N,
            "M": self.M,
            "L": self.K,
            "lr": self.lr
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))
        torch.save(self.model.state_dict(), os.path.join(model_path, model_wts_fname))


    @classmethod
    def load(cls, model_path): 
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        classifier = cls(**model_params)
        classifier.model.load_state_dict(torch.load( os.path.join(model_path, model_wts_fname)))        
        return classifier


def get_data_based_model_params(X):
    '''
    returns a dictionary with N: number of users and M = number of items
    This assumes that the given numpy array (X) has users by id in first column, 
    and items by id in 2nd column. the ids must be 0 to N-1 and 0 to M-1 for users and items.
    '''
    N = int(X[:, 0].max() + 1)
    M = int(X[:, 1].max() + 1)
    return {"N": N, "M": M}


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    try:
        model = Recommender.load(model_path)
    except:
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def save_training_history(history, f_path): 
    with open( os.path.join(f_path, history_fname), mode='w') as f:
        f.write( json.dumps(history, indent=2) )