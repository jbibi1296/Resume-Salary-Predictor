from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

def linear_regression(X,y):

    model = LinearRegression()

    model.fit(X,y)

    return model

def lasso(X,y):

    model = LassoCV(cv = 5)

    model.fit(X,y)

    return model

def ridge(X,y):

    model = RidgeCV()

    model.fit(X,y)

    return model

def random_forest(X,y):
    
    model = RandomForestRegressor(n_estimators = 200,max_depth = 350,
                           criterion='mse', n_jobs=2,
                           min_samples_leaf=3)

    model.fit(X,y)
    
    return model

def gradient_boost(X,y):
    
    model = GradientBoostingRegressor(learning_rate = 0.3, loss = 'ls',
                               max_depth = 200)
    
    model.fit(X,y)
    
    return model

def get_best_epoch(X_train,y_train,X_test,y_test):
    
    nn = Sequential()
    nn.add(Dense(128, activation = 'relu',input_shape=(X_train.shape[1],)))
    nn.add(Dense(32, activation = 'tanh'))
    nn.add(Dense(32, activation = 'relu'))
    nn.add(Dense(64, activation = 'relu'))
    nn.add(Dense(32, activation = 'relu'))
    nn.add(Dense(1, activation = 'linear'))
    nn.compile(optimizer = 'adam', loss = 'mse')
    history = nn.fit(X_train,y_train, 
                        validation_data=(X_test,y_test), 
                        epochs=200, 
                        verbose=0,
                       batch_size =None)
    epoch_df = pd.DataFrame(data = history.history['val_loss'])
    minimum_loss = epoch_df.describe().loc['min'][0]
    best_epoch = epoch_df[epoch_df[0] == minimum_loss].index[0]
    print(f"The best epoch is {best_epoch} with a minimum loss of {minimum_loss}")
    return best_epoch

def neural_net(X_train,y_train,X_test,y_test,best_epoch):
    nn = Sequential()
    nn.add(Dense(128, activation = 'relu',input_shape=(X_train.shape[1],)))
    nn.add(Dense(32, activation = 'tanh'))
    nn.add(Dense(32, activation = 'relu'))
    nn.add(Dense(64, activation = 'relu'))
    nn.add(Dense(32, activation = 'relu'))
    nn.add(Dense(1, activation = 'linear'))
    nn.compile(optimizer = 'adam', loss = 'mse')

    nn.fit(X_train,y_train, 
                        validation_data=(X_test,y_test), 
                        epochs=best_epoch+1, 
                        verbose=0,
                       batch_size =None)
    return nn