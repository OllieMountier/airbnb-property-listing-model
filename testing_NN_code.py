#%%

import torch
import pandas as pd
from itertools import product
import time
import os
import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import nn 
import yaml
import joblib
import json
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, accuracy_score, r2_score

import torch.nn.functional as f
from torchmetrics import Accuracy
from pathlib import Path

dataf = pd.read_csv('clean_tabular_data.csv', usecols=['beds', 'bathrooms', 'Price_Night', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count' ], delimiter=',')

class airbnbnightlyregressiondataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.dataframe = dataframe
    def __getitem__(self, index):
        data_row = self.dataframe.iloc[index]
        features = torch.FloatTensor(data_row.drop(labels='Price_Night').values)
        label = float(data_row['Price_Night'])
        return (features, label)
    
    def __len__(self):
        return len(self.dataframe)

nn_dataset = airbnbnightlyregressiondataset(dataf)

train_data, test_data = random_split(nn_dataset, [664, 166])
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
train_data, val_data = random_split(train_data, [498, 166])
val_loader = DataLoader(val_data, batch_size=166, shuffle= True)   

def generate_nn_configs():
    hyperparameters = {'optimiser': ['SGD'], 
                      'learning_rate': [0.00001, 0.0001, 0.001, 0.01],
                      'momentum': [0.5, 0.9, 0.95, 0.99]}
    hyperparameter_list = []
    keys, values = zip(*hyperparameters.items())
    for j in product(*values):
        hyperparameter_list.append(list(j))
    return hyperparameter_list

class LinearRegression(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(9, 27),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(27, 18),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(18, 1))
        
    def forward(self, features):
        return self.layers(features)
    
def train(model, configuration, train_loader = train_loader, val_loader = val_loader, test_loader = test_loader, epochs = 10):
    i = 0
    train_batch_idx = 0
    writer = SummaryWriter()
    start_training_time = time.time()
    number_of_predictions = 0
    total_prediction_time = 0
    best_r2_score = 0
    for l in configuration:
        current_params = configuration[i]
        optimisertype = getattr(torch.optim, current_params[0])
        lr = current_params[1]
        momentum = current_params[2]
        optimiser = optimisertype(model.parameters(), lr=lr, momentum = momentum)
        for epoch in range(epochs):
            for batch in train_loader:
                features, labels = batch
                labels = labels.type(torch.FloatTensor)
                start_prediction_time = time.time()
                prediction = model(features).reshape(-1)
                end_prediction_time = time.time()
                total_prediction_time += (end_prediction_time - start_prediction_time)
                number_of_predictions += 1
                loss = f.mse_loss(prediction, labels)
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                writer.add_scalar('training loss', loss, train_batch_idx)
                train_batch_idx += 1
        loaders = {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader}
        for loadername, loader in loaders.items():
            load_score = scoring_model(model, loader)
            score_per_set = {'loader': loadername,
                              'rmse': float(load_score[0]),
                             'r2_score': load_score[1]}
            if loadername == 'test_loader' and list(score_per_set.values())[2] > best_r2_score:
                    best_nn_metrics = {'loader': loadername,
                                     'rmse': float(load_score[0]),
                                     'r2_score': load_score[1]}
                    best_r2_score = list(score_per_set.values())[2]
                    best_model = model
                    best_parameters = {'optimiser': optimiser, 
                                       'learning rate': lr, 
                                       'momentum': momentum}
                    
    end_training_time = time.time()
    training_duration = round((end_training_time - start_training_time), 2), 'seconds'
    inference_latency = round(((total_prediction_time/number_of_predictions) * 1000000), 4), 'microseconds'
    best_nn_metrics['training duration'] = training_duration
    best_nn_metrics['inference latency'] = inference_latency
    time_trained = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_final = [best_model, best_parameters, best_nn_metrics]
    idx = 0
    writer = SummaryWriter()
    model_to_test = best_model
    optimiser = list(best_parameters.values())[0]
    for epoch in range(epochs):
        for batch in train_loader:
                features, labels = batch
                labels = labels.type(torch.FloatTensor)
                prediction = model_to_test(features).reshape(-1)
                loss = f.mse_loss(prediction, labels)
                loss.backward()
                r2_scores = r2_score(labels.detach().numpy(), prediction.detach().numpy())
                optimiser.step()
                optimiser.zero_grad()
                writer.add_scalar('best r2 score', r2_scores, idx)
                idx += 1
    save_model(time_trained, 'Neural-Network', 'regression', model, best_parameters, best_nn_metrics)




  

def is_pytorch_module(obj):
    return isinstance(obj, nn.Module)

def scoring_model(model, loader):
    for batches in loader:
        features, label = batches
        label = label.type(torch.FloatTensor)
        prediction = model(features).reshape(-1)
        RMSE_loss = torch.sqrt(f.mse_loss(prediction, label))
        r2_scores = r2_score(label.detach().numpy(), prediction.detach().numpy())
        return RMSE_loss, r2_scores

# def get_nn_config():
#     with open('nn_config.yaml', 'r') as file:
#         config = yaml.safe_load(file)
#         return config  
    
def save_model(time_trained, modeltype, modelname, model, hyperparameters, score, folder='C:/Users/ollie/AiCore/VSCode/Projects/airbnb-property-listing-model/models'):
    is_pytorch = is_pytorch_module(model)
    if is_pytorch == True:
        folder_path = 'C:/Users/ollie/AiCore/VSCode/Projects/airbnb-property-listing-model/models/Neural-Network/Regression/'
        model_direc = time_trained
        full_path = os.path.join(folder_path, model_direc)
        os.makedirs(full_path)
        filename = 'model.pt'
        file = Path(full_path, filename)
        torch.save(model.state_dict(), file)
        hypfilename = 'hyperparameters.json'
        hypfile = Path(full_path, hypfilename)
        opt = list(hyperparameters.values())[0]
        torch.save(opt.state_dict(), hypfile)
        with open(Path(full_path, 'metrics.json'), 'w') as file3:
            json.dump(score, file3)
    else:
        orig_path = folder
        model_type = modeltype
        model_path = os.path.join(orig_path, model_type)
        try:
            os.makedirs(model_path)
        except FileExistsError:
            model_direc = modelname
            full_path = os.path.join(model_path, model_direc)
            os.makedirs(full_path)
            filename = 'model.joblib'
            file = Path(full_path, filename)
            joblib.dump(model, file)
            with open(Path(full_path, 'hyperparameters.json'), 'w') as file2:
                json.dump(hyperparameters, file2)
            with open(Path(full_path, 'metrics.json'), 'w') as file3:
                json.dump(score, file3)

def find_best_nn():
    config = generate_nn_configs()
    model = LinearRegression(config)
    train(model, config)

find_best_nn()


# %%
