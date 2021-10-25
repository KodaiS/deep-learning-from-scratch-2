# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params  # embedレイヤの重みW
        self.grads = self.embed.grads  # embed.backward()で更新される
        self.cache = None

    def forward(self, h, idx):
        # h: 中間層からの出力
        # idx: 正解の単語の単語ID (要素数は batch_size)
        target_W = self.embed.forward(idx)  # 正解の単語IDに対応した行の重みを抽出
        out = np.sum(target_W * h, axis=1)  # 中間層出力と正解単語の重みの要素ごとの積を取ったのちに行ごとに和を計算．hと各行の内積を実現．

        self.cache = (h, target_W)  # backwardで使うため
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)  # 1次元で渡されたdoutを2次元に変換

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)  # embedレイヤの勾配を更新
        dh = dout * target_W
        return dh


class UnigramSampler:
    """
    コーパス中の単語の確率分布に従って，ランダムに単語をサンプリングする
    corpus: コーパス．単語IDのリスト．
    power: 確率分布を累乗．1より小．稀な単語の確率を少し上げる．
    sample_size: いくつサンプリングするか？
    """
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # コーパスの単語の出現回数をカウント
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        # 単語の出現確率を計算
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)  # 確率を保持する行列
        for i in range(vocab_size):
            self.word_p[i] = counts[i]  # 出現回数を入れる

        self.word_p = np.power(self.word_p, power)  # 累乗
        self.word_p /= np.sum(self.word_p)  # 確率に変換

    def get_negative_sample(self, target):
        """
        ターゲット以外の単語IDを単語の出現確率に従ってランダムに抽出する
        target: ターゲットの単語IDをnumpy array（ミニバッチ）で渡す．
        """
        batch_size = target.shape[0]

        if not GPU:
            # ミニバッチ数 x サンプリングする数の行列
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()  # 配列 p を初期化
                target_idx = target[i]  # i 番目のターゲットの単語ID
                p[target_idx] = 0  # ターゲットの確率を0にする
                p /= p.sum()  # ターゲットを除いて確率を再計算
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)  # 重複なしでサンプリング
        else:
            # GPU(cupy）で計算するときは、速度を優先
            # 負例にターゲットが含まれるケースがある
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size), replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例のフォワード
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 負例のフォワード
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh
