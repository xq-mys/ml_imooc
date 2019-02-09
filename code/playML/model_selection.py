import numpy as np

"""将数据X和y按照test_ratio分割成X_train,X_test,y_train,y_test"""
def train_test_split(X, y, test_ratio=0.2, seed=None):

    # 判断输入数据正确性
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ratio must be valid"

    if seed:
        np.random.seed(seed)

    # 随机排列X数据集的索引
    shuffled_indexs = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio) # 测试数量
    test_indexs = shuffled_indexs[:test_size] # 测试索引
    train_indexs = shuffled_indexs[test_size:] # 训练索引

    # 训练数据集
    X_train = X[train_indexs]
    y_train = y[train_indexs]

    # 测试数据集
    X_test = X[test_indexs]
    y_test = y[test_indexs]

    return X_train, X_test, y_train, y_test