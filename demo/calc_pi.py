import numpy as np

if __name__ == '__main__':
    # 计算pi
    print np.sqrt(6 * np.sum(1 / np.arange(1, 10000, dtype=np.float) ** 2))
