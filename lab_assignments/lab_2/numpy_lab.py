import numpy as np

def n_size_ndarray_creation(n, dtype=np.int):
    X = np.arange(n,dtype = dtype)
    return X*X

#print(n_size_ndarray_creation(10, float))


def zero_or_one_or_empty_ndarray(shape, type=0, dtype=np.int):
    if type==0:
        return np.zeros(shape, dtype=dtype)
    elif type ==1:
        return np.ones(shape, dtype=dtype)
    else:
        return np.empty(shape, dtype=dtype)

#print(zero_or_one_or_empty_ndarray(shape=(2,2), type=1))
#print(zero_or_one_or_empty_ndarray(shape=(3,3), type=99))


def change_shape_of_ndarray(X, n_row):
    if n_row==1:
        return X.flatten()
    else:
        return X.reshape(n_row, -1)

# X = np.ones((32,32), dtype=np.int)
# print(change_shape_of_ndarray(X, 1))
#print(change_shape_of_ndarray(X, 512))


def concat_ndarray(X_1, X_2, axis):
    if X_1.ndim ==1:
        X_1 = X_1.reshape(1, -1)
    if X_2.ndim ==1:
        X_2 = X_2.reshape(1, -1)
    row_1, column_1 = X_1.shape
    row_2, column_2 = X_2.shape
    if axis==0:
        if column_1 ==column_2:
            return np.vstack((X_1, X_2))
        return False
    else:
        if row_1 == row_2:
            return np.hstack((X_1, X_2))
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
    print(X.shape)
    if axis == 99:
        return (X - X.mean()) / X.std()
    if axis == 0:
        print(X.mean(axis=axis).shape, X.mean(axis=axis).shape)
        return (X - X.mean(axis=axis)) / X.std(axis=axis)
    if axis == 1:
        print(X.mean(axis=axis).shape, X.mean(axis=axis).reshape(n_row, -1).shape)
        return (X - X.mean(axis=axis).reshape(n_row, -1)) / X.std(axis=axis).reshape(n_row, -1)

# X = np.arange(12, dtype=np.float32).reshape(6,2)
# print(normalize_ndarray(X))
# print(normalize_ndarray(X, 1))
# print(normalize_ndarray(X, 0))


def save_ndarray(X, filename="test.npy"):
    np.savetxt(filename, X)
# X = np.arange(32, dtype=np.float32).reshape(4, -1)
# filename = "test.npy"
# save_ndarray(X, filename) #test.npy 파일이 생성됨


def boolean_index(X, condition):
    return np.where(eval(str("X")+condition))

# X = np.arange(32, dtype=np.float32).reshape(4, -1)
# print(boolean_index(X, "== 3"))
# X = np.arange(32, dtype=np.float32)
# print(boolean_index(X, "> 6"))

def find_nearest_value(X, target_value):
    return X[np.argmin(np.abs(X-target_value))] #np.abs(): 절대값함수

# X = np.random.uniform(0, 1, 100)
# target_value = 0.3
# print(find_nearest_value(X, target_value))


def get_n_largest_values(X, n):
    return np.sort(X)[-n:-1]

# X = np.random.uniform(0, 1, 100)
# print(get_n_largest_values(X, 3))
# print(get_n_largest_values(X, 5))
