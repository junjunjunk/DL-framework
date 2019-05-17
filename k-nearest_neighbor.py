
'''
k-nearest neighbor (K近傍法)
入力パターンに近いk個の学習パターンを取り上げ，最も多数を占めたカテゴリを入力パターンのカテゴリとする。

1.入力パターンと全ての学習パターンとの距離を計算する。
2.距離の昇順に学習パターンをソートし、上位k個で最も出現しているカテゴリを出力とする。

※距離…コサイン距離やユークリッド距離など。
'''

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class_num = 10

class KNN:
    #入力パターン、ラベル、距離関数
    def __init__(self, x, y, func):
        self.train_x = x
        self.train_y = y
        self.distance_func = func
    
    #入力パターンに対して予測ラベルを返す
    def prediction(self, X, k):
        #全ての入力パターンと全ての学習データとの距離を計算
        distance_matrix = self.distance_func(X, self.train_x) 
        #距離のに学習パターンをソートする
        sort_index = np.argsort(distance_matrix, axis = 1)
        nearest_k = sort_index[:,:k]
        nearest_k = sort_index[:,:k] 
        labels = self.train_y[nearest_k] 
        label_num = np.sum(np.eye(class_num)[labels], axis=1) 
        Y = np.argmax(label_num, axis=1) 
        return Y
    
    #予測データと正解データを用いてaccuracyを計算する
    def get_accuracy(self, pred, real, eval_func=accuracy_score):
        accuracy = eval_func(pred, real)
        return accuracy
    
    # 最適なkを見つけるためにkを変化させて予測を行い，最も性能が高いkを返す
    def find_k(self, val_x, val_y, k_list):
        score_list = []
        #for k in tqdm(k_list): 
        for k in k_list:
            pred = self.prediction(val_x, k)
            accuracy = self.get_accuracy(pred, val_y)
            print('k：{0}, accuracy：{1:.5f}'.format(k,accuracy))
            score_list.append(accuracy)

        top_ind = np.argmax(score_list)
        best_k = k_list[top_ind]
        print('best k : {0}, val score : {1:.5f}'.format(best_k,score_list[top_ind]))
        return best_k