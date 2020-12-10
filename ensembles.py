import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 bootstrap_size=None, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_size = feature_subsample_size
        self.bootstrap_size = bootstrap_size
        self.kwargs = trees_parameters

        self.score = []
        self.ensemble = []
        self.feature_indexs = None
        self.obj_indexs = None

    def prepocessing_params(self, X):
        if self.bootstrap_size is None:
            self.bootstrap_size = X.shape[0]
        else:
            self.feature_size = self.bootstrap_size*X.shape[0]

        if self.feature_size is None:
            self.feature_size = X.shape[1]//3
        else:
            self.feature_size = self.feature_size*X.shape[1]

        self.obj_indexs = [np.random.choice(X.shape[0], X.shape[0], replace=True)
                           for i in range(self.n_estimators)]

        self.feature_indexs = [np.random.choice(X.shape[1], self.feature_size, replace=False)
                               for i in range(self.n_estimators)]

    def write_score(self, X_val, y_val, scorer):
        if X_val is not None and y_val is not None and scorer is not None:
            self.score.append(scorer(self.predict(X_val), y_val))

    def fit(self, X, y, X_val=None, y_val=None, scorer=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects           
        """
        self.prepocessing_params(X)

        for i in range(self.n_estimators):
            f_ind = self.feature_indexs[i]
            obj_ind = self.feature_indexs[i]

            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.kwargs)
            model.fit(X[obj_ind, f_ind], y[obj_ind])

            self.ensemble.append(model)
            self.write_score(X_val, y_val, scorer)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        y_pred = np.zeros(X.shape[0])
        for f, m in zip(self.feature_indexs, self.ensemble):
            y_pred += m.predict(X[:, f])
        return y_pred/X.shape[0]


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_size = feature_subsample_size
        self.kwargs = trees_parameters

        self.score = []
        self.ensemble = []
        self.lambdas = []
        self.feature_indexs = None

    def prepocessing_params(self, X):
        if self.feature_size is None:
            self.feature_size = X.shape[1]//3
        else:
            self.feature_size = self.feature_size*X.shape[1]

        self.feature_indexs = [np.random.choice(X.shape[1], self.feature_size, replace=False)
                               for i in range(self.n_estimators)]

    def write_score(self, X_val, y_val, scorer):
        if X_val is not None and y_val is not None and scorer is not None:
            self.score.append(scorer(self.predict(X_val), y_val))

    def fit(self, X, y, X_val=None, y_val=None, scorer=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        self.prepocessing_params(X)

        model = DummyRegressor(strategy="mean")
        model.fit(X, y)

        self.lambdas.append(1)
        self.ensemble.append(model)
        self.feature_indexs.append(None)
        self.write_score(X_val, y_val, scorer)

        for i in range(self.n_estimators-1):
            f_ind = self.feature_indexs[i]

            y_pred_base = self.predict(X)
            grad = 2*(y_pred_base-y)

            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.kwargs)
            model.fit(X[:, f_ind], -grad)
            y_pred_new = model.predict(X[:, f_ind])
            l = minimize_scalar(lambda l: mean_squared_error(y, y_pred_base+l*y_pred_new))

            self.lambdas.append(self.learning_rate*l)
            self.ensemble.append(model)
            self.write_score(X_val, y_val, scorer)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        y_pred = np.zeros(X.shape[0])
        for l, f_ind, m in zip(self.lambdas, self.feature_indexs, self.ensemble):
            y_pred += l*m.predict(X[:, f_ind])
        return y_pred
