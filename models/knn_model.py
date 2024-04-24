from sklearn.neighbors import KNeighborsRegressor


class KnnModel:

    def __init__(self, n_neighbors=3):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
