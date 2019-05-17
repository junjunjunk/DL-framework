#距離関数のまとめ

import numpy as np

#コサイン距離
def cosine_distance(X, Y):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        Z = 1 - np.dot(X, Y.T)
        return Z

#ユークリッド距離
def euclidean_distance(X, Y):
    X = X[:, np.newaxis, :]
    X = np.hstack([X for i in range(Y.shape[0])])
    Y = Y[np.newaxis, :, :]
    Y = np.vstack([Y for i in range(X.shape[0])])
    Z = np.linalg.norm(X - Y, axis=2)
    return Z