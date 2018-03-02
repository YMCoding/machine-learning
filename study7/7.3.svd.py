#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint


def restore1(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量、保留K个
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    # 前K个不断的累加
    for k in range(K):
        # 所有行都要，只取第k列，转成m行1列的列向量
        uk = u[:, k].reshape(m, 1)
        # 1行n列的行向量
        vk = v[k].reshape(1, n)
        # dot 生成一个m行n列的矩阵
        a += sigma[k] * np.dot(uk, vk)
    a[a < 0] = 0
    a[a > 255] = 255
    # a = a.clip(0, 255)
    return np.rint(a).astype('uint8')


def restore2(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K + 1):
        for i in range(m):
            a[i] += sigma[k] * u[i][k] * v[k]
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')


if __name__ == "__main__":
    # 读图片
    A = Image.open("7.son.png", 'r')
    # 输出文件夹
    output_path = r'.\Pic'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # 图片像素点转成数组，由RGB三原色组成的三个二维数组
    a = np.array(A)
    # 整个图片的红色像素组
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])
    # 整个图片的绿色像素组
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])
    # 整个图片的蓝色像素组
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])
    plt.figure(figsize=(10, 10), facecolor='w')
    # 为了显示中文
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # 保留前50个奇异值，在后面就舍掉
    K = 50
    for k in range(1, K + 1):
        print k
        R = restore1(sigma_r, u_r, v_r, k)
        G = restore1(sigma_g, u_g, v_g, k)
        B = restore1(sigma_b, u_b, v_b, k)
        # 三原色叠加，每一个元素各自叠加
        I = np.stack((R, G, B), 2)
        # 保存图像
        Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
        if k <= 12:
            plt.subplot(3, 4, k)
            plt.imshow(I)
            plt.axis('off')
            plt.title(u'奇异值个数：%d' % k)
    plt.suptitle(u'SVD与图像分解', fontsize=20)
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.9)
    plt.show()
