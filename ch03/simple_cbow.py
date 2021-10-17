# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size  # V: 語彙数, H: エンコード後の単語ベクトルの次元数

        # 重みの初期化
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # レイヤの生成
        self.in_layer0 = MatMul(W_in)  # 入力層->中間層の全結合層
        self.in_layer1 = MatMul(W_in)  # in_layerはコンテキストと同じ数
        self.out_layer = MatMul(W_out)  # 中間層->出力層の全結合層
        self.loss_layer = SoftmaxWithLoss()  # 出力層->softmax(確率)->cross entropy(損失)

        # すべての重みと勾配をリストにまとめる
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params  # レイヤの重み，バイアス
            self.grads += layer.grads  # レイヤの勾配

        # メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5  # 二つのin_layerの平均が中間層
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        # 各レイヤのbackwardメソッドを順に呼び出す
        # レイヤのbackwardメソッドが実行されるたび，インスタンス変数の勾配が更新される
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
