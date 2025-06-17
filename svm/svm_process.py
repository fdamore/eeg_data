import sys
import time
import numpy as np
sys.path.append('.')


import dataproc.preprocessing as dp
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

filename = '/home/francesco/git/esn_rpy/data/signal_eeg.csv'

df_origin = pd.read_csv(filename)
nrow = df_origin.shape[0]

print(f'Origin shape: {df_origin.shape}')

f_rate = 0.2
df = pd.read_csv(filename)[:int(np.ceil(nrow * f_rate))]

print(f'Resampled df: {df.shape}')

#clean and preprocessing the data and apply feature reduction
df_clean =  dp.clean_data(df.copy(), scale=False)
X, y = dp.split_data_target(df_clean)


print(f'Shape of the data: {X.shape}')
print(f'Shape of the label: {y.shape}')
print(f'Number of classes: {len(np.unique(y))}')


#apply feature reduction
data_reduced, model_ca = dp.lda_process(X, y, n_components=None)

#require a specific variance threshold
required_var = 0.95

X_reduced = dp.get_data_for_variance(required_variance=required_var, ca=model_ca, X=data_reduced)
print(f'Number of components required to reach {required_var*100}% variance: {X_reduced.shape[1]}')
print(f'Shape of the reduced data: {X_reduced.shape}')

params_grid = {
    'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
    'gamma': ['auto', 'scale'],
    'kernel':['rbf'],
    'decision_function_shape': ['ovr', 'ovo'] 
    }

#Create the GridSearchCV object
nfolds = 10  # Number of cross-validation folds
grid = GridSearchCV(SVC(), params_grid, verbose=1, cv=nfolds, n_jobs=1, scoring='accuracy')

#get time
t_start = time.time()

#Fit the data with the best possible parameters
grid_clf = grid.fit(X_reduced, y)

#get time training
t_training = time.time()

#Print the best estimator with it's parameters
print(f'Best paramenter: {grid.best_params_}')
print(f'Time training: {t_training - t_start} seconds')
print(f'Best score {grid.best_score_}')
print(f'Results: {grid.cv_results_.keys()}')






