import numpy as np


class LinearRegression(object):
    def __init__(self, fit_intercept=True, copy_X=True):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

        self._coef = None
        self._intercept = None
        self._new_X = None

    def fit(self, X, y):
        #fit_intercept -> 절편값을 만들어 준다.
        #normal equation을 가지고 weight값을 찾아낸다.
        self._new_X = np.array(X) #X가 numpy array가 아닌 list 형태로 들어올 수도 있으므로 np.arrya(X)를 해준다.
        y = y.reshape(-1, 1) #계산의 편의를 위해서 2 dim array로 변경

        if self.fit_intercept:
            intercept_vector = np.ones([len(self._new_X), 1])
            self._new_X = np.concatenate((intercept_vector, self._new_X), axis=1)

        weights = np.linalg.inv(self._new_X.T.dot(self._new_X)).dot(self._new_X.T.dot(y)).flatten()

        if self.fit_intercept:
            self._coef = weights[1:]
            self._intercept = weights[0]
        else:
            self._coef = weights


    def predict(self, X):
        #y의 hat을 반환
        test_X = np.array(X)

        if self.fit_intercept:
            intercept_vector = np.ones([len(test_X), 1])
            test_X = np.concatenate((intercept_vector, test_X), axis=1)

            weights = np.concatenate(([self._intercept], self._coef), axis=0)
        else:
            weights = self._coef

        return test_X.dot(weights) # y = Xw


    @property
    def coef(self): #w[1] ~ w[n]까지의 값을 반환하면 된다.
        return self._coef

    @property
    def intercept(self):
        return self._intercept
