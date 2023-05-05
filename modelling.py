#%%
from tabular_data import load_airbnb

from itertools import product
import numpy as np
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

#y_test_pred = model.predict(X_test)
#y_train_pred = model.predict(X_train)

#test_score = r2_score(y_test, y_test_pred)
#train_score = r2_score(y_train, y_train_pred)


#test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
#train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))

hyperparameter_dict = {SGDRegressor(): {'learning_rate': ['constant', 'optimal', 'invscaling'], 'loss': ['squared_error', 'huber', 'epsilon_insensitive'], 'penalty': ['l1', 'l2', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'max_iter': [5000, 8000, 10000, 15000, 20000]}, 
                       DecisionTreeRegressor(): {'splitter': ['best', 'random'], 'min_samples_split': [5, 10, 15, 20, 30], 'min_samples_leaf': [5, 10, 15, 20], 'max_leaf_nodes': [None, 20, 40, 60, 80, 100]}, RandomForestRegressor(): {'n_estimators': [50, 100, 200, 400],'min_samples_split': [5, 10, 15],        'min_samples_leaf': [8, 10, 12], 'max_depth': [None, 3, 5, 7]}, GradientBoostingRegressor(): {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'], 'min_samples_split': [5, 10, 15], 'min_samples_leaf': [8, 12, 16], 'max_depth': [None, 3, 5, 7]}}

models = [SGDRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]

def custom_tune_regression_model_hyperparameters(hyperparameters, model = models, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, X_val = X_val, y_val = y_val):
    for h in hyperparameters:
        keys, values = zip(*hyperparameters[h].items())
        print(keys, values)
        best_score = -100000
        for j in product(*values):
            g = dict(zip(keys, j))
            test_model = h
            test_model = test_model.fit(X_train, y_train)
            y_pred = test_model.predict(X_train)
            score = test_model.score(X_train, y_train)
            if score > best_score:
                best_model = test_model
                best_score = score
                best_parameters = g
                rmse = mean_squared_error(y_train, y_pred, squared=False)
        print('Best score for', best_model, 'is: ', best_score)
        #print('Best parameters are: ', best_parameters)
        #print('Best rmse is: ' ,rmse)

def tune_regression_model_hyperparameters(hyperparameters, model = models, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, X_val = X_val, y_val = y_val) :
    for h in hyperparameters:
        param_grid = dict(hyperparameters[h].items())
        gridsearch = GridSearchCV(h, param_grid)
        gridsearch = gridsearch.fit(X_train, y_train)
        best_params = gridsearch.best_params_
        accuracy = gridsearch.best_score_
        print(h, best_params, accuracy)


# def save_model(model, hyperparameters, score, folder='C:/Users/ollie/AiCore/VSCode/Projects/airbnb-property-listing-model/models/regression/'):
#     filename = 'model.joblib'
#     file = Path(folder, filename)
#     joblib.dump(model, file)
#     with open(Path(folder, 'hyperparameters.json'), 'w') as file2:
#         json.dump(hyperparameters, file2)
#     with open(Path(folder, 'metrics.json'), 'w') as file3:
#         json.dump(score, file3)

custom_tune_regression_model_hyperparameters(hyperparameter_dict)
tune_regression_model_hyperparameters(hyperparameter_dict)

#tes = tune_regression_model_hyperparameters()
#print(tes)
#save_model(end_model[2], end_model[1], end_model[0])




# param_grid = {'n_estimators': [50, 100, 200, 400],
#               'min_samples_split': [5, 10, 15], 
#               'min_samples_leaf': [8, 10, 12], 
#               'max_depth': [None, 3, 5, 7]}

# param_grid2 = {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
#                'min_samples_split': [5, 10, 15], 
#                'min_samples_leaf': [8, 12, 16], 
#                'max_depth': [None, 3, 5, 7]}

# gridsearch = GridSearchCV(GradientBoostingRegressor(), param_grid).fit(X_train, y_train)
# best_gs_score = gridsearch.best_score_
# best_gs_params = gridsearch.best_params_
# print(best_gs_score, best_gs_params)


#%%-
