#%%
from typing import Any
from tabular_data import load_airbnb, load_airbnb_cat

from itertools import product
import numpy as np
import joblib
from pathlib import Path
import json, os
import yaml
import torch
import pandas as pd
import time
import datetime

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import nn 

import torch.nn.functional as f
from torchmetrics import Accuracy

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, scale, OneHotEncoder, LabelEncoder

from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier,GradientBoostingClassifier

from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, accuracy_score, r2_score
import matplotlib.pyplot as plt

features_label_tuple = load_airbnb('clean_tabular_data.csv')
airbnb_df = features_label_tuple[0]
airbnb_df['beds'] = features_label_tuple[1]
airbnb_df_tbe = pd.DataFrame(airbnb_df['Category'])

one = OneHotEncoder(sparse=False)
airbnb_df_ft = one.fit_transform(airbnb_df_tbe)
one_df = pd.DataFrame(airbnb_df_ft, columns=one.get_feature_names_out())
dataframe_encoded = pd.concat([airbnb_df, one_df], axis = 1).drop(['Category'], axis = 1)
features = dataframe_encoded.loc[:, dataframe_encoded.columns !='beds']
labels = dataframe_encoded['beds']

# features_category_tuple = load_airbnb_cat('clean_tabular_data.csv')
# airbnb_df_nocat = features_category_tuple[0]
# airbnb_df_nocat['Category'] = features_category_tuple[1]
# class_features = airbnb_df_nocat.loc[:, airbnb_df_nocat.columns !='Category']
# class_label = airbnb_df_nocat['Category']

# X_price = price_features
# y_price = price_label
# X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size = 0.2, random_state=1)
# X_price_train, X_price_val, y_price_train, y_price_val = train_test_split(X_price_train, y_price_train, test_size=0.25, random_state = 1)
# scrm = StandardScaler()
# scrm.fit(X_price_train)
# X_price_train = scrm.transform(X_price_train)
# X_price_test = scrm.transform(X_price_test)
# X_price_val = scrm.transform(X_price_val)

# X_cat = class_features
# y_cat = class_label
# X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(X_cat, y_cat, test_size = 0.2, random_state=1)
# X_cat_train, X_cat_val, y_cat_train, y_cat_val = train_test_split(X_cat_train, y_cat_train, test_size=0.25, random_state = 1)
# sccm = StandardScaler()
# sccm.fit(X_cat_train)
# X_cat_train = sccm.transform(X_cat_train)
# X_cat_test = sccm.transform(X_cat_test)
# X_cat_val = sccm.transform(X_cat_val)

np.random.seed(2)

X = features
scrm = StandardScaler()
X.loc[:, X.columns !='Category'] = scrm.fit_transform(X.loc[:, X.columns !='Category'])
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state = 1)


regression_dict = {SGDRegressor: 
                   {'learning_rate': ['constant', 'optimal', 'invscaling'],
                    'loss': ['squared_error', 'huber', 'epsilon_insensitive'], 
                    'penalty': ['l1', 'l2', 'elasticnet'], 
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                    'max_iter': [10000]}, 
                    DecisionTreeRegressor:{ 
                     'min_samples_split': [10, 20, 30, 40], 
                     'min_samples_leaf': [5, 10, 15, 20], 
                     'max_leaf_nodes': [None, 20, 40, 60, 80, 100]}, 
                     RandomForestRegressor: 
                     {'n_estimators': [50, 100, 200, 400],
                      'min_samples_split': [10, 20, 30, 40],
                      'min_samples_leaf': [5, 10, 15, 20]}, 
                      GradientBoostingRegressor: 
                      {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'], 
                       'min_samples_split': [10, 20, 30, 40], 
                       'min_samples_leaf': [5, 10, 15, 20]}}


classification_dict = {LogisticRegression: 
                        {'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                        'max_iter': [10000], 
                        'C': [0.001, 0.01, 0.1, 1, 10, 100]},
                        DecisionTreeClassifier: 
                        {'min_samples_split': [10, 20, 30, 40],
                        'min_samples_leaf': [5, 10, 15, 20]},
                        RandomForestClassifier: 
                        {'min_samples_split': [10, 20, 30, 40],
                        'n_estimators': [50, 100, 150],
                        'min_samples_leaf': [5, 10, 15, 20]},
                        GradientBoostingClassifier:
                        {'n_estimators':[50, 100, 150],
                        'min_samples_split': [10, 20, 30, 40],
                        'min_samples_leaf': [5, 10, 15, 20]}}

regression_model_names = ['SGDRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor']
classification_model_names = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier','GradientBoostingRegressor']

def baseline_model(model, X_train, y_train, X_test, y_test):
    model = model()
    baseline_train = model.fit(X_train, y_train)
    base_train_pred = baseline_train.predict(X_train)
    base_test_pred = baseline_train.predict(X_test)
    return (baseline_train, base_train_pred, base_test_pred)

def baseline_model_scores(train_model, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred):
    train_model = train_model
    if train_model == 'regression':
        train_accuracy = train_model.score(X_train, y_train)
        test_accuracy = train_model.score(X_test, y_test)
        train_rmse = mean_squared_error(y_train, y_train_pred, squared = False)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared = False)
#         scores_dict = {'model': train_model, 
#                        'train_scores': {'accuracy': train_accuracy,
#                                         'rmse': train_rmse},
#                         'test_score': {'accuracy': test_accuracy,
#                                        'rmse': test_rmse}}
#     elif train_model == 'classification':
#         train_f1 = f1_score(y_train, y_train_pred, average = None)
#         train_precision = precision_score(y_train, y_train_pred, average = None)
#         train_recall = recall_score(y_train, y_train_pred, average = None)    
#         train_accuracy = train_model.score(X_train, y_train)
#         test_f1 = f1_score(y_test, y_test_pred, average = None)
#         test_precision = precision_score(y_test, y_test_pred, average = None)
#         test_recall = recall_score(y_test, y_test_pred, average = None)    
#         test_accuracy = train_model.score(X_test, y_test)
#         scores_dict={'model': train_model,
#                      'train_scores': {'f1_score': train_f1, 
#                                       'precision': train_precision, 
#                                       'recall': train_recall,
#                                       'validation_accuracy': train_accuracy}, 
#                      'test_scores': {'f1_score': test_f1, 
#                                       'precision': test_precision, 
#                                       'recall': test_recall,
#                                       'validation_accuracy': test_accuracy}}
#     return scores_dict


# custom_tuning_mh = list(regression_dict.items())[0]
# def custom_tune_regression_model_hyperparameters(model, hyperparameters, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, X_val = X_val, y_val = y_val):
#         best_score = -100000
#         keys, values = zip(*hyperparameters.items())
#         for j in product(*values):
#             g = dict(zip(keys, j))
#             test_model = model(**g)
#             test_model.fit(X_train, y_train)
#             y_pred = test_model.predict(X_val)
#             score = test_model.score(X_val, y_val)
#             if score > best_score:
#                 best_model = test_model
#                 best_score = score
#                 best_parameters = g
#                 rmse = mean_squared_error(y_val, y_pred, squared=False)
#                 metrics = {rmse, best_score}
#         model_stats = [best_model, best_parameters, metrics]
#         print(model_stats) 
        
def tune_regression_model_hyperparameters(hyperparameters, modelnames, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, X_val = X_val, y_val = y_val) :
    i = 0
    writer = SummaryWriter()
    for hp in hyperparameters:
        modelname = modelnames[i]
        param_grid = dict(hyperparameters[hp].items())
        rm = hp()
        rm_gridsearch = GridSearchCV(rm, param_grid).fit(X_train, y_train)
        y_pred = rm_gridsearch.predict(X_val)
        best_model = rm_gridsearch.best_estimator_
        best_score = rm_gridsearch.best_score_
        best_hyperparameters = rm_gridsearch.best_params_
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        best_metrics = {'validation_accuracy': best_score,
                        'best_rmse': rmse}
        rm_model_list = [best_model, best_hyperparameters, best_metrics]
        # plt.plot(score, rm_gridsearch.cv_results_[list('params')])
        # plt.legend()
        # plt.xlabel('mean_test_Score')
        # plt.ylabel('parameters')
        # plt.show()
        # time_trained = time.time()
        #save_model(time_trained, 'regression', modelname, rm_model_list[0], rm_model_list[1], rm_model_list[2])
        i+=1

def tune_classification_model_hyperparameters(hyperparameters, modelnames, X_train = X_train, X_val = X_val, y_train = y_train, y_val = y_val):
    i = 0
    for hp in hyperparameters:
        modelname = modelnames[i]
        param_grid = dict(hyperparameters[hp].items())
        lgm_model = hp()
        gridsearch_model = GridSearchCV(lgm_model, param_grid).fit(X_train, y_train)
        y_pred = gridsearch_model.predict(X_val)
        print(y_pred)
        best_lgm_hyperparameters = gridsearch_model.best_params_
        f1 = f1_score(y_val, y_pred, average = None, zero_division=1)
        precision = precision_score(y_val, y_pred, average = None)
        recall = recall_score(y_val, y_pred, average = None)    
        accuracy = gridsearch_model.score(X_val, y_val)
        best_lgm_metrics = {'f1_score': f1.tolist(), 
                        'precision': precision.tolist(), 
                        'recall': recall.tolist(),
                        'validation_accuracy': accuracy}
        model_list = [gridsearch_model.best_estimator_, best_lgm_hyperparameters, best_lgm_metrics]
        print(model_list)
        time_trained = time.time()
        #save_model(time_trained, 'classification', modelname, model_list[0], model_list[1], model_list[2])
        i+=1

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
        with open(Path(full_path, 'hyperparameters.json'), 'w') as file2:
            json.dump(hyperparameters, file2)
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
 
def find_best_model(task_folder, X_train, y_train, X_val, y_val):
    best_score = -10000
    u = 0
    basedir = 'C:/Users/ollie/AiCore/VSCode/Projects/airbnb-property-listing-model/models'
    model_name = task_folder
    rootdir = os.path.join(basedir, model_name)
    for dirs in os.listdir(rootdir):
        model = 'model.joblib'
        model_path = os.path.join(rootdir, dirs)
        absolute_path = os.path.join(model_path, model)
        model_to_score = joblib.load(absolute_path).fit(X_train, y_train)
        models_score = model_to_score.score(X_val, y_val)
        print(models_score)
        y_pred = model_to_score.predict(X_val)
        if models_score > best_score:
            best_model = model_to_score
            best_hyperparameters = model_to_score.get_params()
            if model_name == 'regression':
                best_score = models_score
                model_rmse = mean_squared_error(y_val, y_pred, squared = 'False')
                best_model_metrics = {'Validation accuracy': best_score,
                                      'RMSE': model_rmse}
            elif model_name == 'classification':
                best_model_f1 = f1_score(y_val, y_pred, average = None)
                best_model_precision = precision_score(y_val, y_val, average = None)
                best_model_recall = recall_score(y_val, y_pred, average = None, zero_division = 1)
                best_model_accuracy = best_model.score(X_val, y_val)
                best_model_metrics = {'f1_score': best_model_f1, 
                                      'precision': best_model_precision, 
                                      'recall': best_model_recall,
                                     'accuracy': best_model_accuracy}
    return (best_model, best_hyperparameters, best_model_metrics)
    
def evaluate_all_models():
    #custom_tune_regression_model_hyperparameters(custom_tuning_mh[0], custom_tuning_mh[1])
    tune_regression_model_hyperparameters(regression_dict, regression_model_names)
    #tune_classification_model_hyperparameters(classification_dict, classification_model_names)    

dataf = pd.read_csv('clean_tabular_data.csv', usecols=['beds', 'bathrooms', 'Price_Night', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count' ], delimiter=',')

class airbnbnightlyregressiondataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.dataframe = dataframe
    def __getitem__(self, index):
        data_row = self.dataframe.iloc[index]
        features = torch.FloatTensor(data_row.drop(labels='beds').values)
        label = float(data_row['beds'])
        return (features, label)
    
    def __len__(self):
        return len(self.dataframe)

nn_dataset = airbnbnightlyregressiondataset(dataframe_encoded)

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
        self.layers = torch.nn.Sequential(torch.nn.Linear(14, 32),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(32, 16),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(16, 1))
        
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
                print(prediction)
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
    print(model_final)
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

def get_nn_config():
    with open('nn_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        return config  
    
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

if __name__ == '__main__':
    #evaluate_all_models()
    reg_model = find_best_model('regression', X_train, y_train ,X_val, y_val)
    class_model = find_best_model('classification', X_train, y_train ,X_val, y_val)
    #find_best_nn()
    #modelreg = best_regression_model
modelreg = reg_model[0]
modelclass = class_model[0]

modelreg = modelreg.fit(X_train, y_train)
reg_pred = modelreg.predict(X_test)
rmse = mean_squared_error(y_test, reg_pred, squared=False)
        
modelclass = modelclass.fit(X_train, y_train)
class_pred = modelclass.predict(X_test)
accuracy = modelclass.score(X_test, y_test)

plt.scatter(reg_pred, y_test)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

   
    

#%%