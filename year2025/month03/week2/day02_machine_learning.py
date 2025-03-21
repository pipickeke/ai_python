from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz



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

def knn_iris_gscv():
    """
    用KNN算法对鸢尾花进行分类，添加网格搜索和交叉验证
    :return:
    """
    iris = load_iris()
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, random_state=22)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_train)

    estimator = KNeighborsClassifier()

    param_dict = {"n_neighbors": [1,3,5,7,9]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)

    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print("预测值：\n", y_predict)
    ans = list()
    for x,y in zip(y_predict, y_test):
        if x == y:
            ans.append(True)
        else:
            ans.append(False)
    print("比较结果：\n", ans)

    # score = estimator.score(x_test, y_test)
    # print("准确率：\n", score)
    print("最佳参数：\n", estimator.best_params_)
    print("最佳结果：\n", estimator.best_score_)
    print("最佳估计器: \n", estimator.best_estimator_)
    print("交叉验证结果: \n", estimator.cv_results_)



def nb_news():
    """
    朴素贝叶斯算法对新闻进行分类
    :return:
    """
    news = fetch_20newsgroups(subset='all')

    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print("预测值 y_predict: \n", y_predict)
    print("真实值和预测值比较：\n", y_test == y_predict)

    print("准确率：\n", estimator.score(x_test, y_test))



def decision_iris():
    """
    用决策树对鸢尾花进行分类
    :return:
    """
    iris = load_iris()
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, random_state=22)

    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print("预测值：\n", y_predict)

    print("准确率： \n", estimator.score(x_test, y_test))

    # 可视化决策树
    # http://webgraphviz.com/ 可视化
    export_graphviz(estimator, out_file="iris_tree.dot",
                    feature_names=iris.feature_names)



if __name__ == '__main__':
    # knn_iris()
    # knn_iris_gscv()
    # nb_news()
    decision_iris()