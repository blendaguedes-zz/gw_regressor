import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#functio used to normalize the data
min_max_scaler = preprocessing.MinMaxScaler()

#read the `masses` file
masses = pd.read_csv("masses", delimiter=' ')

#drop the third unnecessary column
masses = masses.drop(columns='NaN')

#normalizing masses
m_scaled = min_max_scaler.fit_transform(masses.values)
masses_normalized = pd.DataFrame(m_scaled, columns=['m1', 'm2'])

#read the histograms
waves_histogram = pd.read_csv("different_full_size_waves_hist", delimiter=' ', header=None)

#drop the last unecessary column
waves_histogram = waves_histogram.iloc[:, :-1]

#normalizing histograms
w_scaled = min_max_scaler.fit_transform(waves_histogram.values)
waves_histogram_nomalized = pd.DataFrame(w_scaled)


''' Here we start the Machine Learning solutions'''

#paramenters of grid search
p_activation = ['identity', 'logistic', 'tanh', 'relu']
p_solver = ['sgd', 'adam', 'lbfgs']
p_learning_rate = ['constant', 'invscaling', 'adaptive']
p_max_iter = [200]
p_random_state = [5]
p_verbose = [False]
p_alpha = [0.0001, 0.001, 0.01,0.1,10,100]

#creating paramenters dictionary
parameters ={'activation': p_activation,
             'solver': p_solver,
             'alpha': p_alpha,
             'learning_rate' : p_learning_rate,
             'max_iter': p_max_iter,
             'random_state': p_random_state,
             'verbose': p_verbose}

'''scores are the metrics that will point into the best combination of parameters
    both of scores used here are better when closer to 0 (zero)
    we will use 2 (two) differents ways to look at our result'''

scores = ['neg_mean_absolute_error', 'neg_mean_squared_error']

mlp = MLPRegressor()


for score in scores:
    '''here we create the grid search
        cv = number of cross validation
        refit = what will point the refit, in this case will be the score
        n_jobs = parallel jobs'''
    reg = GridSearchCV(mlp, parameters, cv=5, scoring=score, refit=score, n_jobs=4)

    '''here we start the search for the best combination of parameters
        the cross validation lib already splits  between trainig and validation'''
    grid_result = reg.fit(waves_histogram_nomalized, masses_normalized)

    '''result of mean, score and paramenters'''
    s_mae = grid_result.cv_results_['mean_test_score']
    stds_mae = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    print('Best estimador:', reg.best_estimator_, '\n')
    print('Best score:', reg.best_score_, '\n')
    result = []
    """ summarize results """
    result.append("----------------------------------------------------------" + "\n")
    result.append("----------------------------------------------------------" + "\n")
    result.append("Best: %f using %s" % (grid_result.best_score_, grid_result.best_estimator_) + "\n")
    result.append("----------------------------------------------------------" + "\n")


    for mean_mae, stdev_mae, param in zip(s_mae, stds_mae, params):
        print("mean mae=%f (std=%f) with: %r" % (mean_mae, stdev_mae, param) + "\n")
        result.append("mean mae=%f (std=%f) with: %r" % (mean_mae, stdev_mae, param) + "\n")

        print("****************************************************" + "\n")
        result.append("****************************************************" + "\n")

    '''here we write the result in a file, the file will be aumatically generated and be on your project folder'''
    file_result = open("result_cv_"+score, 'a')
    file_result.writelines(result)
    file_result.close()

print('THE END')