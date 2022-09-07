import json
import sys

import numpy as np, pandas as pd
import os
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MODEL_NAME = "recommender_base_matrix_factorizer_in_pytorch"

model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self, N, M, K):
        super().__init__()
        self.u_embedding = nn.Embedding(N, K)  # shape => (N, 1, K)
        self.m_embedding = nn.Embedding(M, K)  # shape => (N, 1, K)
        self.u_bias = nn.Embedding(N, 1)  # shape => (N, 1, 1)
        self.m_bias = nn.Embedding(M, 1)  # shape => (N, 1, 1)

    def forward(self, x):
        u, m = torch.split(x, [1, 1], dim=1)
        t = torch.squeeze(torch.sum(self.u_embedding(u) * self.m_embedding(m), dim=2))
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


class Recommender():

    def __init__(self, N, M, K=10, batch_size=256, **kwargs):
        self.N = N
        self.M = M
        self.K = K
        self.batch_size = batch_size

        self.model = Net(self.N, self.M, self.K)
        self.model.to(device)
        # print(self.model.get_num_parameters())
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()


    def fit(self, train_X, train_y, valid_X, valid_y, epochs=25, verbose=1, use_early_stopping=True, patience=3):
        
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_X), torch.tensor(train_y))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        
        if valid_X is not None and valid_y is not None:
            valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_X), torch.tensor(valid_y))
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        else: 
            valid_dataset = valid_loader = None
            
        best_loss = 1e7
        min_epochs = 1

        print_every = 1000  # steps
        
        train_losses = []
        for e in range(epochs):
            self.model.train()
            Ntrain = len(train_loader)
            for step, data in enumerate(train_loader):
                x, y = data[0].to(device), data[1].to(device).float()
                # Make predictions
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)                
                # Prevent accumulation of gradients
                self.optimizer.zero_grad()
                # backprop
                loss.backward()
                # Weight Update: w <-- w - lr * gradient
                self.optimizer.step()

                current_loss = loss.item()
                
                if verbose == 1 and step % print_every == 0:
                    print(f'Epoch: {e+1}/{epochs}, Step:{step+1}/{Ntrain}; loss: {np.round(current_loss, 5)}')
            
            print("done epoch")
            if use_early_stopping:
                # Early stopping
                if valid_loader is not None: 
                    current_loss = get_loss(self.model, device, valid_loader, self.criterion)   
                
                if current_loss < best_loss:
                    trigger_times = 0
                    best_loss = current_loss
                else:
                    trigger_times += 1
                    if trigger_times >= patience and e >= min_epochs:
                        if verbose == 1: print('Early stopping!')
                        return train_losses
            
                if verbose == 1:
                    print(f'Epoch: {e+1}/{epochs}, loss: {np.round(current_loss, 5)}')
                    
        return train_losses
    
    
    def __call__(self, X):
        return self.predict(X)
    

    def predict(self, X):
        X = torch.LongTensor(X).to(device)
        preds = self.model(X).detach().cpu().numpy().reshape(-1, 1)
        return preds
    

    def summary(self):
        print(self.model)
        

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return the loss and metrics"""
        test = torch.utils.data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
        test_loader = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)        
        loss = get_loss(self.model, device, test_loader, self.criterion)
        return loss
    

    def save(self, model_path):
        model_params = {
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "batch_size": self.batch_size,
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))
        torch.save(self.model.state_dict(), os.path.join(model_path, model_wts_fname))


    @classmethod
    def load(ml, model_path):
        # print("found the model weights? 1")
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        # print("found the model weights 2")
        mf = ml(**model_params)
        # print("found the model weight333?")
        mf.model.load_state_dict(torch.load(os.path.join(model_path, model_wts_fname)))
        return mf


def get_loss(model, device, data_loader, loss_function):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input = data[0].to(device)
            label = data[1].to(device)
            output = model(input.view(input.shape[0], -1))
            loss = loss_function(output, label)
            loss_total += loss.item()
    return loss_total / len(data_loader)


def get_data_based_model_params(train_X, valid_X):
    '''
    returns a dictionary with N: number of users and M = number of items
    This assumes that the given numpy array (X) has users by id in first column, 
    and items by id in 2nd column. 
    The ids must be contiguous i.e. 0 to N-1 and 0 to M-1 for users and items.
    '''
    N_train = int(train_X[:, 0].max() + 1)
    M_train = int(train_X[:, 1].max() + 1)
    if valid_X is not None: 
        N_valid = int(valid_X[:, 0].max() + 1)
        M_valid = int(valid_X[:, 1].max() + 1)
    else: N_valid = M_valid = 0
        
    return {"N": max(N_train, N_valid), "M": max(M_train, M_valid)}


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
    with open(os.path.join(f_path, history_fname), mode='w') as f:
        f.write(json.dumps(history, indent=2))
