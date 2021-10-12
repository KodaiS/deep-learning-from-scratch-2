# coding: utf-8
from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W):
        self.params = [W]  # 重みをインスタンス変数で保持
        self.grads = [np.zeros_like(W)]  # 勾配をインスタンス変数で保持．Wと同じ形状．
        self.x = None  # xをインスタンス変数で保持する．順伝播でセットする．

    def forward(self, x):
        W, = self.params  # 重みを参照
        out = np.dot(x, W)  # 積和を計算
        self.x = x  # 入力xは逆伝播に使うためインスタンス変数で保持
        return out  # 下流にoutを伝播

    def backward(self, dout):
        W, = self.params  # 重みを参照
        dx = np.dot(dout, W.T)  # xでの偏微分を計算．
        dW = np.dot(self.x.T, dout)  # Wでの偏微分を計算
        self.grads[0][...] = dW  # メモリ位置を固定してdWをコピー．勾配が更新される．
        return dx  # 下流にはdxを伝播


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]  # 重みとバイアスをリストでインスタンス変数に保持
        self.grads = [np.zeros_like(W), np.zeros_like(b)]  # 重みとバイアスの勾配
        self.x = None  # 順伝播で入力を保持する

    def forward(self, x):
        W, b = self.params  # パラメータを参照
        out = np.dot(x, W) + b  # Affine変換
        self.x = x  # 入力を保持
        return out

    def backward(self, dout):
        W, b = self.params  # パラメータを参照
        dx = np.dot(dout, W.T)  # xでの偏微分を計算
        dW = np.dot(self.x.T, dout)  # Wでの偏微分を計算
        db = np.sum(dout, axis=0)  # bでの偏微分を計算

        self.grads[0][...] = dW  # Wの勾配
        self.grads[1][...] = db  # bの勾配
        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []  # パラメータと勾配
        self.out = None  # 出力

    def forward(self, x):
        self.out = softmax(x)  # ソフトマックス関数を計算
        return self.out  # ソフトマックス関数の結果を返す

    def backward(self, dout):
        dx = self.out * dout  # xでの偏微分
        sumdx = np.sum(dx, axis=1, keepdims=True)  
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoidの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


class Dropout:
    '''
    http://arxiv.org/abs/1207.0580
    '''
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


# 4.1 word2vecの改良
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None  # 抽出する行のインデックスを保持する変数

    def forward(self, idx):
        W, = self.params
        self.idx = idx  # ここでidxを設定
        out = W[idx]  # 重みから指定したidxの行だけを取り出す
        return out

    def backward(self, dout):
        # gradの全要素を0にする
        dW, = self.grads
        dW[...] = 0
        # forwardで抽出したインデックスに対応したdoutで勾配を更新
        # ミニバッチに同じインデックスが複数あった場合は足し合わせる
        if GPU:
            np.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None
