from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier



def knn_iris():
    """
    KNN算法对鸢尾花进行分类
    :return:
    """
    iris = load_iris()
    x_train, x_test, y_train, y_test =\
        train_test_split(iris.data, iris.target, random_state=6)

    transter = StandardScaler()
    x_train = transter.fit_transform(x_train)
    x_test = transter.transform(x_train)

    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    #模型评估
    #（1）直接比对真实值和预测值
    #（2）计算准确率

    y_predict = estimator.predict(x_test)
    print("预测值：\n", y_predict)
    print("真实值和预测值比对：\n", y_predict == y_test)

    # score = estimator.score(x_test, y_test)
    # print("准确率：\n", score)



if __name__ == '__main__':
    knn_iris()