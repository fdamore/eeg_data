import sys
import time
import numpy as np
sys.path.append('.')


import dataproc.preprocessing as dp
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

filename = '/home/francesco/git/esn_rpy/data/signal_eeg.csv'

df_origin = pd.read_csv(filename)
nrow = df_origin.shape[0]

print(f'Origin shape: {df_origin.shape}')

f_rate = 0.5
df = pd.read_csv(filename)[:int(np.ceil(nrow * f_rate))]

print(f'Resampled df: {df.shape}')

#clean and preprocessing the data and apply feature reduction
df_clean =  dp.clean_data(df.copy(), scale=False)
X, y = dp.split_data_target(df_clean)


print(f'Shape of the data: {X.shape}')
print(f'Shape of the label: {y.shape}')
print(f'Number of classes: {len(np.unique(y))}')


#method for feature extraction
feature_exractor_method_pca = 'pca'
feature_exractor_method_lda = 'lda'

feature_exractor_method = 'lda'''

if feature_exractor_method is None:
    # Create a pipeline with scaling and SVC
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC())
    ])

    # Update parameter grid to use pipeline step names
    params_grid = {
        'svc__C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
        'svc__gamma': ['auto', 'scale'],
        'svc__kernel': ['rbf'],
        'svc__decision_function_shape': ['ovr', 'ovo']
    }

if feature_exractor_method == feature_exractor_method_pca:
    from sklearn.decomposition import PCA
    # Create a pipeline with PCA, scaling, and SVC
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),        
        ('svc', SVC())
    ])

    # Update parameter grid to include PCA parameters
    params_grid = {
        'pca__n_components': [5,6,7,8,9,10],
        'svc__C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
        'svc__gamma': ['auto', 'scale'],
        'svc__kernel': ['rbf'],
        'svc__decision_function_shape': ['ovr', 'ovo']
    }

if feature_exractor_method == feature_exractor_method_lda:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    # Create a pipeline with LDA, scaling, and SVC
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LDA()),        
        ('svc', SVC())
    ])

    # Update parameter grid to include LDA parameters
    params_grid = {
        'lda__n_components': [5,6,7,8,9,10],
        'svc__C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
        'svc__gamma': ['auto', 'scale'],
        'svc__kernel': ['rbf'],
        'svc__decision_function_shape': ['ovr', 'ovo']
    }


#Create the GridSearchCV object
nfolds = 2  # Number of cross-validation folds
grid = GridSearchCV(pipe, params_grid, verbose=1, cv=nfolds, n_jobs=1, scoring='accuracy')

#get time
t_start = time.time()

#Fit the data with the best possible parameters
grid_clf = grid.fit(X, y)

#get time training
t_training = time.time()

#Print the best estimator with it's parameters
print(f'Time training: {t_training - t_start} seconds')
print("Best estimator:\n", grid.best_estimator_)
print("Best score (mean cross-validated accuracy):", grid.best_score_)
print("Best parameters:", grid.best_params_)

print("Grid scores on development set:")

means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print(f"{mean:.3f} (+/-{std * 2:.3f}) for {params}")

print("All cv_results_ keys:", list(grid.cv_results_.keys()))
print(f'Feature extractor method: {feature_exractor_method}')
print(f'Number of folds (nfolds): {nfolds}')




