# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 加载文件 按照delimiter 分割，skiprows跳过前几行 usecols读那列 unpack散开
    stock_max, stock_min, stock_close, stock_amount = np.loadtxt('7.SH600000.txt', delimiter='\t', skiprows=2, usecols=(2, 3, 4, 5), unpack=True)
    N = 100
    stock_close = stock_close[:N]
    # 收盘价格
    print stock_close
    # 取五天，每天的权值是0.2
    n = 5
    weight = np.ones(n)
    weight /= weight.sum()
    print weight
    # 做卷积
    stock_sma = np.convolve(stock_close, weight, mode='valid')  # simple moving average

    # 新的权值卷积 指数的移动平均线
    weight = np.linspace(1, 0, n)
    weight = np.exp(weight)
    weight /= weight.sum()
    print weight
    stock_ema = np.convolve(stock_close, weight, mode='valid')  # exponential moving average

    # 多项式拟合
    t = np.arange(n-1, N)
    poly = np.polyfit(t, stock_ema, 10)
    print poly
    stock_ema_hat = np.polyval(poly, t)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(np.arange(N), stock_close, 'ro-', linewidth=2, label=u'原始收盘价')
    t = np.arange(n-1, N)
    plt.plot(t, stock_sma, 'b-', linewidth=2, label=u'简单移动平均线')
    plt.plot(t, stock_ema, 'g-', linewidth=2, label=u'指数移动平均线')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(np.arange(N), stock_close, 'r-', linewidth=1, label=u'原始收盘价')
    plt.plot(t, stock_ema, 'g-', linewidth=2, label=u'指数移动平均线')
    plt.plot(t, stock_ema_hat, 'm-', linewidth=3, label=u'指数移动平均线估计')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
