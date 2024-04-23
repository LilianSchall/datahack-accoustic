from sklearn import linear_model

class LinearModel:
    def __init__(self):
        self.model = linear_model.LinearRegression()
    
    def fit(X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(X_test):
        self.model.predict(X_test)
    