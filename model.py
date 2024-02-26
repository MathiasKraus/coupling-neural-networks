import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from copy import deepcopy

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import torch
from torch import nn
    
class LLM_coupled:
    """
    A LogitLeafModel with coupled linear leaf models. This means each Linear Leaf Model has a loss from the other LMs.
    This model already features a leaf model distance weighting for the additional loss.
    """

    def __init__(self, reg_strength=0, lr=0.004, epochs=10000,
                 device='cpu', verbose=False):
        self.dt = None
        self.lm_dict = {}
        self.leaf_idx = None
        self.reg_strength = reg_strength
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.verbose = verbose

    def fit(self, X, y, clusters):
        self.leaf_idx = clusters

        glob_lr = LinearRegression()
        glob_lr.fit(X, y)
        
        for idx in np.unique(self.leaf_idx):
            self.lm_dict[idx] = nn.Linear(X.shape[1], 1).to(self.device)
            
            self.set_nn_parameter_data(self.lm_dict[idx], 'weight', torch.from_numpy(np.array([glob_lr.coef_])).float().to(self.device))
            self.set_nn_parameter_data(self.lm_dict[idx], 'bias', torch.from_numpy(np.array([glob_lr.intercept_])).float().to(self.device))

        self.fit_coupled_lms(X, y)

    def set_nn_parameter_data(self, layer, parameter_name, new_data):
        param = getattr(layer, parameter_name)
        param.data = new_data

    def fit_coupled_lms(self, X, y):
        X = torch.from_numpy(np.array(X)).float().to(self.device)
        y = torch.from_numpy(np.array(y)).float().to(self.device)

        # get the parameter of each leaf model, so that we can calculate the coupled loss later
        params = []
        for idx in np.unique(self.leaf_idx):
            params.extend(list(self.lm_dict[idx].parameters()))

        optimizer = torch.optim.Adam(params, lr=self.lr)

        leaf_distances = 1.

        if self.verbose:
            pbar = tqdm(range(self.epochs))
        loss_history = []
        for i in range(self.epochs):
            loss = torch.tensor(0.0).to(self.device)
            optimizer.zero_grad()
            for idx in np.unique(self.leaf_idx):
                X_leaf = X[self.leaf_idx == idx].to(self.device)
                y_leaf = y[self.leaf_idx == idx].to(self.device)
                # losses[idx] = torch.tensor(0.0).to(self.device)
                # if the leaf model is a dummy classifier, we don't need to train it and add any loss, since it is already trained
                out = self.lm_dict[idx](X_leaf)

                loss += nn.MSELoss()(out.squeeze(), y_leaf)

            if self.reg_strength > 0:
                for idx1 in np.unique(self.leaf_idx):
                    for idx2 in np.unique(self.leaf_idx):
                        for params1, params2 in zip(self.lm_dict[idx1].parameters(), self.lm_dict[idx2].parameters()):

                            loss += self.reg_strength * (torch.linalg.norm(params1 - params2) ** 2)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if i % 10 == 0 and self.verbose:
                # calculate not the average loss but the average weighted loss with respect to the correct loss func
                pbar.set_description(
                    f"Loss: {loss.item()}")  # {np.mean([losses[idx].item() for idx in np.unique(self.leaf_idx)])}")
                pbar.update(10)
                pbar.refresh()

        if self.verbose:
            # if multiple loss functions: calculate average weighted! final loss
            print("Final Loss: %.4f" % loss.item())
            print("Training finished.")
            if self.verbose:
                pbar.close()

    def predict(self, X, clusters):
        leaf_idx = clusters

        X = torch.from_numpy(np.array(X)).float().to(self.device)

        pred = torch.zeros(len(X)).to(self.device)
        for idx in np.unique(self.leaf_idx):
            X_leaf = X[leaf_idx == idx].to(self.device)

            pred[leaf_idx == idx] = self.lm_dict[idx](X_leaf).squeeze()

        return pred.cpu().detach().numpy()