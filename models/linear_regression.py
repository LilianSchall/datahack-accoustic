from sklearn import linear_model

class LinearRegression:
    def __init__(self):
        self.model = linear_model.LinearRegression()
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)