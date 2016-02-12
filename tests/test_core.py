import numpy as np
from dprocess.core import DataCache
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

def test_hello():
    
    dc = DataCache('./dc')
    
    # TODO: how to parametrize this kind of stuff?
    def make_data_set():
        X = np.arange(20)[np.newaxis].T
        y = np.arange(20)
        return X, y
    
    def split(prev):
        X, y = prev
        return train_test_split(X, y)
        
    def square(prev):
        X, y = prev
        
        return X**2, y

    def linear_fit(dataset, C=1.0):
        X_train, X_test, y_train, y_test = dataset
        lr = LinearRegression(C)
        lr.fit(X_train, y_train)
        
        return lr, lr.score(X_test, y_test)
    
    dataset = dc.step(make_data_set).step(split)
    lr, score = dc.record(linear_fit, dataset)
    
    dataset2 = dc.step(make_data_set).step(square).step(split)
    lr, score = dc.record(linear_fit, dataset2)
    
    print dc.summary()
