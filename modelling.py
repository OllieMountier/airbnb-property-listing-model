#%%
from tabular_data import load_airbnb

from itertools import product
import numpy as np
import joblib
from pathlib import Path
import json
import os
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale


features_label_tuple = load_airbnb('clean_tabular_data.csv')
airbnb_df = features_label_tuple[0]
airbnb_df['Price_Night'] = features_label_tuple[1]
features = airbnb_df.loc[:, airbnb_df.columns !='Price_Night']
label = airbnb_df['Price_Night']

X = features
X= scale(X)
y = label
y = scale(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state = 1)

np.random.seed(2)

hyperparameter_dict = {SGDRegressor: {'learning_rate': ['constant', 'optimal', 'invscaling'], 'loss':            ['squared_error', 'huber', 'epsilon_insensitive'], 'penalty': ['l1', 'l2', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'max_iter': [5000, 8000, 10000, 15000, 20000]}, 
                       DecisionTreeRegressor: {'splitter': ['best', 'random'], 'min_samples_split': [40, 60, 80, 100], 'min_samples_leaf': [25, 50, 75, 100], 'max_leaf_nodes': [None, 20, 40, 60, 80, 100]}, RandomForestRegressor: {'n_estimators': [50, 100, 200, 400],'min_samples_split': [5, 10, 15],        'min_samples_leaf': [25, 50, 75], 'max_depth': [None, 3, 5, 7]}, GradientBoostingRegressor: {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'], 'min_samples_split': [75, 100, 125], 'min_samples_leaf': [100, 150, 200], 'max_depth': [None, 3, 5, 7]}}

models = ['SGDRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor']

def custom_tune_regression_model_hyperparameters(hyperparameters, modelnames = models, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, X_val = X_val, y_val = y_val):
    i = 0
    while i < 1:
        for h in hyperparameters:
            modelname = modelnames[i]
            best_score = -100000
            keys, values = zip(*hyperparameters[h].items())
            for j in product(*values):
                g = dict(zip(keys, j))
                test_model = h(**g)
                test_model.fit(X_train, y_train)
                y_pred = test_model.predict(X_train)
                score = test_model.score(X_train, y_train)
                if score > best_score:
                    best_model = test_model
                    best_score = score
                    best_parameters = g
                    rmse = mean_squared_error(y_train, y_pred, squared=False)
                    metrics = {rmse, best_score}
            model_stats = [best_model, best_parameters, metrics]
            
            save_model(modelname, model_stats[0], model_stats[1], model_stats[2])
            i+=1
    
        
        

def tune_regression_model_hyperparameters(hyperparameters, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, X_val = X_val, y_val = y_val) :
    for h in hyperparameters:
        param_grid = dict(hyperparameters[h].items())
        h = h()
        gridsearch = GridSearchCV(h, param_grid)
        gridsearch = gridsearch.fit(X_train, y_train)
        y_pred = gridsearch.predict(X_train)
        best_params = gridsearch.best_params_
        accuracy = gridsearch.best_score_
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        print(h, best_params, accuracy, rmse)


def save_model(modelname, model, hyperparameters, score, folder='C:/Users/ollie/AiCore/VSCode/Projects/airbnb-property-listing-model/models/regression/'):
    folder = folder
    direc = modelname
    path1 = os.path.join(folder, direc)
    os.makedirs(path1)
    filename = 'model.joblib'
    file = Path(path1, filename)
    joblib.dump(model, file)
    with open(Path(path1, 'hyperparameters.json'), 'w') as file2:
        json.dump(hyperparameters, file2)
    with open(Path(path1, 'metrics.json'), 'w') as file3:
        json.dump(score, file3)



def evaluate_all_models():
    custom_tune_regression_model_hyperparameters(hyperparameter_dict)
    tune_regression_model_hyperparameters(hyperparameter_dict)

def find_best_model():
    best_score = -10000
    u = 0
    while u < 4:
        folder='C:/Users/ollie/AiCore/VSCode/Projects/airbnb-property-listing-model/models/regression/'
        direc = models[u]
        path2 = os.path.join(folder, direc)
        model67 = 'model.joblib'
        path3 = os.path.join(path2, model67)
        model_to_score = joblib.load(path3)
        model_to_score = model_to_score.fit(X_train, y_train)
        models_score = model_to_score.score(X_train, y_train)
        y_pred = model_to_score.predict(X_train)
        if models_score > best_score:
            best_model = model_to_score
            best_params = dict(model_to_score.get_params())
            performance_metrics = {'rmse': mean_squared_error(y_train, y_pred, squared=False), 'r2_score': models_score}
            best_score = models_score
        u+=1
    return best_model, best_params, performance_metrics



    

if __name__ == '__main__':
    evaluate_all_models()
    best_model = find_best_model()
    
    
    
    
    
    
#%%