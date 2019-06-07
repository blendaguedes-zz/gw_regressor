from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import cross_val_score

min_max_scaler = preprocessing.MinMaxScaler()

'''same code as in grid_search.py to get and normalize data'''
masses = pd.read_csv("masses", delimiter=' ')
masses = masses.drop(columns='NaN')
m_scaled = min_max_scaler.fit_transform(masses.values)
masses_normalized = pd.DataFrame(m_scaled, columns=['m1', 'm2'])

waves_histogram = pd.read_csv("different_full_size_waves_hist", delimiter=' ', header=None)
waves_histogram = waves_histogram.iloc[:, :-1]
w_scaled = min_max_scaler.fit_transform(waves_histogram.values)
waves_histogram_nomalized = pd.DataFrame(w_scaled)


X_train, X_test, y_train, y_test = train_test_split(masses_normalized.values, waves_histogram_nomalized.values, test_size=0.4, random_state=0)

'''MLP Regressor with the chosen parameters on grid_search.py
    Here we used Mean Absolute Error as a exemple, but you can add Mean Squared Error too, as your wish'''

'''regressor = MLPRegressor(activation='logistic', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=10, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False).fit(X_train, y_train)'''

clf = MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=5, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False).fit(X_train, y_train)
print(clf.score(X_test, y_test))
#scores = cross_val_score(clf, waves_histogram_nomalized, masses_normalized, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



