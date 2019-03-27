# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:37:59 2018

@author: m.nagano
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
import copy

def main():
    #dataset = datasets.load_iris()
    #features = dataset.data
    data = np.loadtxt("000.txt")
    
    # 主成分分析する
    pca = PCA(n_components=2)
    pca.fit(data)

    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(data)
    
    # 入力をプロットする
    x = np.arange(0,len(data),1)
    plt.figure()
    for i in range(len(data[0])):
        plt.plot(x,data[:,i])
    plt.savefig("alldata.png")

    # 主成分をプロットする
    plt.figure()
    plt.plot(transformed[:,0],transformed[:,1],"-o")
    #for label in np.unique(targets):
    #    plt.scatter(transformed[targets == label, 0],
    #                transformed[targets == label, 1])
    plt.title('Latent Space')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.savefig("LatentSpace_dim2.png")
    # 主成分の寄与率を出力する
    #print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
    #print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

    # グラフを表示する
    plt.show()
    np.savetxt("LatentSpace_dim2.txt",transformed)

if __name__ == '__main__':
    main()
