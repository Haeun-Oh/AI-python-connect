import numpy as np

def n_size_ndarray_creation(n, dtype=np.int):
    X = np.arange(n,dtype = dtype)
    return np.arane(range(n**2), dtype=dtype).reshape(n, n)

#print(n_size_ndarray_creation(10, float))


def zero_or_one_or_empty_ndarray(shape, type=0, dtype=np.int):
    if type==0:
        return np.zeros(shape=shape, dtype=dtype)
    if type ==1:
        return np.ones(shape=shape, dtype=dtype)
    if type==99:
        return np.empty(shape=shape, dtype=dtype)

#print(zero_or_one_or_empty_ndarray(shape=(2,2), type=1))
#print(zero_or_one_or_empty_ndarray(shape=(3,3), type=99))


def change_shape_of_ndarray(X, n_row):
    return X.flatten() if n_row==1 else X.reshape(n_row, -1)

# X = np.ones((32,32), dtype=np.int)
# print(change_shape_of_ndarray(X, 1))
#print(change_shape_of_ndarray(X, 512))


def concat_ndarray(X_1, X_2, axis):
    try:
        if X_1.ndim ==1:
            X_1 = X_1.reshape(1, -1)
        if X_2.ndim ==1:
            X_2 = X_2.reshape(1, -1)
        return np.concatenate((X_1, X_2), axis=axis)
    except ValueError as e:
        return False

# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6]])
# #print(concat_ndarray(a, b, 0))
# print(concat_ndarray(a, b, 1))
# a = np.array([1, 2])
# b = np.array([5, 6, 7])
# print(concat_ndarray(a, b, 1))
# print(concat_ndarray(a, b, 0))

def normalize_ndarray(X, axis=99, dtype=np.float32):
    n_row, n_column = X.shape
    if axis == 99:
        X_mean = np.mean(X)
        X_std = np.std(X)
        Z = (X - X_mean) / X_std
    if axis == 0:
        X_mean = np.mean(X, 0).reshape(1, -1)
        X_std = np.std(X, 0).reshape(1, -1)
        print(X_std.shape)
        Z = (X-X_mean)/X_std
    if axis == 1:
        X_mean = np.mean(X, 1).reshape(n_row, -1)
        X_std = np.std(X, 1).reshape(n_row, -1)
        Z = (X - X_mean) / X_std

    return Z

X = np.arange(12, dtype=np.float32).reshape(6,2)
print(normalize_ndarray(X))
print(normalize_ndarray(X, 1))
print(normalize_ndarray(X, 0))
def save_ndarray(X, filename="test.npy"):
    pass


def boolean_index(X, condition):
    condition = eval(str("X")+condition)
    return np.where(condition)


def find_nearest_value(X, target_value):
    return X[np.argmin(np.abs(X-target_value))]


def get_n_largest_values(X, n):
    return X[np.argsort(X[::-1])[:n]]
