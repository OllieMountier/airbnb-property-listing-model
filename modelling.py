#%%
from tabular_data import load_airbnb, load_airbnb_cat

from itertools import product
import numpy as np
import joblib
from pathlib import Path
import json, os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier,GradientBoostingClassifier

from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score

features_label_tuple = load_airbnb('clean_tabular_data.csv')
airbnb_df = features_label_tuple[0]
airbnb_df['Price_Night'] = features_label_tuple[1]
features = airbnb_df.loc[:, airbnb_df.columns !='Price_Night']
label = airbnb_df['Price_Night']

features_category_tuple = load_airbnb_cat('clean_tabular_data.csv')
airbnb_df_nocat = features_category_tuple[0]
airbnb_df_nocat['Category'] = features_category_tuple[1]
class_features = airbnb_df_nocat.loc[:, airbnb_df_nocat.columns !='Category']
class_label = airbnb_df_nocat['Category']

X_price = features
y_price = label
X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size = 0.2, random_state=1)
X_price_train, X_price_val, y_price_train, y_price_val = train_test_split(X_price_train, y_price_train, test_size=0.25, random_state = 1)
scrm = StandardScaler()
scrm.fit(X_price_train)
X_price_train = scrm.transform(X_price_train)
X_price_test = scrm.transform(X_price_test)
X_price_val = scrm.transform(X_price_val)

X_cat = class_features
y_cat = class_label
X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(X_cat, y_cat, test_size = 0.2, random_state=1)
X_cat_train, X_cat_val, y_cat_train, y_cat_val = train_test_split(X_cat_train, y_cat_train, test_size=0.25, random_state = 1)
sccm = StandardScaler()
sccm.fit(X_cat_train)
X_cat_train = sccm.transform(X_cat_train)
X_cat_test = sccm.transform(X_cat_test)
X_cat_val = sccm.transform(X_cat_val)




np.random.seed(2)

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
        scores_dict = {'model': train_model, 
                       'train_scores': {'accuracy': train_accuracy,
                                        'rmse': train_rmse},
                        'test_score': {'accuracy': test_accuracy,
                                       'rmse': test_rmse}}
    elif train_model == 'classification':
        train_f1 = f1_score(y_train, y_train_pred, average = None)
        train_precision = precision_score(y_train, y_train_pred, average = None)
        train_recall = recall_score(y_train, y_train_pred, average = None)    
        train_accuracy = train_model.score(X_train, y_train)
        test_f1 = f1_score(y_test, y_test_pred, average = None)
        test_precision = precision_score(y_test, y_test_pred, average = None)
        test_recall = recall_score(y_test, y_test_pred, average = None)    
        test_accuracy = train_model.score(X_test, y_test)
        scores_dict={'model': train_model,
                     'train_scores': {'f1_score': train_f1, 
                                      'precision': train_precision, 
                                      'recall': train_recall,
                                      'validation_accuracy': train_accuracy}, 
                     'test_scores': {'f1_score': test_f1, 
                                      'precision': test_precision, 
                                      'recall': test_recall,
                                      'validation_accuracy': test_accuracy}}
    return scores_dict


custom_tuning_mh = list(regression_dict.items())[0]
def custom_tune_regression_model_hyperparameters(model, hyperparameters, X_train = X_price_train, y_train = y_price_train, X_test = X_price_test, y_test = y_price_test, X_val = X_price_val, y_val = y_price_val):
        best_score = -100000
        keys, values = zip(*hyperparameters.items())
        for j in product(*values):
            g = dict(zip(keys, j))
            test_model = model(**g)
            test_model.fit(X_train, y_train)
            y_pred = test_model.predict(X_val)
            score = test_model.score(X_val, y_val)
            if score > best_score:
                best_model = test_model
                best_score = score
                best_parameters = g
                rmse = mean_squared_error(y_val, y_pred, squared=False)
                metrics = {rmse, best_score}
        model_stats = [best_model, best_parameters, metrics]
        print(model_stats) 
        
def tune_regression_model_hyperparameters(hyperparameters, modelnames, X_train = X_price_train, y_train = y_price_train, X_test = X_price_test, y_test = y_price_test, X_val = X_price_val, y_val = y_price_val) :
    i = 0
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
        save_model('regression', modelname, rm_model_list[0], rm_model_list[1], rm_model_list[2])
        i+=1

def tune_classification_model_hyperparameters(hyperparameters, modelnames, X_train = X_cat_train, X_val = X_cat_val, y_train = y_cat_train, y_val = y_cat_val):
    i = 0
    for hp in hyperparameters:
        modelname = modelnames[i]
        param_grid = dict(hyperparameters[hp].items())
        lgm_model = hp()
        gridsearch_model = GridSearchCV(lgm_model, param_grid).fit(X_train, y_train)
        y_pred = gridsearch_model.predict(X_val)
        best_lgm_hyperparameters = gridsearch_model.best_params_
        f1 = f1_score(y_val, y_pred, average = None)
        precision = precision_score(y_val, y_pred, average = None)
        recall = recall_score(y_val, y_pred, average = None)    
        accuracy = gridsearch_model.score(X_val, y_val)
        best_lgm_metrics = {'f1_score': f1.tolist(), 
                        'precision': precision.tolist(), 
                        'recall': recall.tolist(),
                        'validation_accuracy': accuracy}
        model_list = [gridsearch_model.best_estimator_, best_lgm_hyperparameters, best_lgm_metrics]
        print(model_list)
        #save_model(modelname, model_list[0], model_list[1], model_list[2])
        i+=1

def save_model(modeltype, modelname, model, hyperparameters, score, folder='C:/Users/ollie/AiCore/VSCode/Projects/airbnb-property-listing-model/models/'):
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
    print(best_model, best_hyperparameters, best_model_metrics)

def evaluate_all_models():
    custom_tune_regression_model_hyperparameters(custom_tuning_mh[0], custom_tuning_mh[1])
    tune_regression_model_hyperparameters(regression_dict, regression_model_names)
    tune_classification_model_hyperparameters(classification_dict, classification_model_names)    

if __name__ == '__main__':
    evaluate_all_models()
    find_best_model()

    
    
    
#%%