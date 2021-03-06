import numpy as np
from math import sqrt
from collections import Counter

def KNN_classify(k, X_train, y_train, x):

    # 保证用户传来的输入是合法的
    assert 1 <= k <= X_train.shape[0], \
		"K must be valid" # k最多为训练集中的样本数量
    assert X_train.shape[0] == y_train.shape[0],\
        "the size of X_train must equal to the size of y_train"  # X_train中样本数量==y_train中样本数量相等
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"

    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]
