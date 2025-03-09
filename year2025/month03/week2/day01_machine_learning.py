from sklearn.datasets import load_iris



def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)
    print("数据集描述：\n", iris['DESCR'])
    print("数据集特征值名称：\n", iris.feature_names)
    print("特征值：\n", iris.data)




if __name__ == '__main__':
    datasets_demo()