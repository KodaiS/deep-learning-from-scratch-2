# coding: utf-8
from common.np import *


def sigmoid(x):
    # 0 - 1の値を取る
    # x = 0のとき0.5，xが大きいほど1に近く，xが小さいほど0に近くなる
    return 1 / (1 + np.exp(-x))


def relu(x):
    # x >= 0 のときはxを返し，x < 0 のときは0を返す
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)  # オーバーフロー対策
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)  # オーバーフロー対策
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
