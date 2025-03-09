from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split




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

    # 数据集的划分
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)


def dict_demo():
    """
    字典特征提取
    :return:
    """
    data = [{"city": "北京", "temperature": 100},{"city": "上海", "temperature": 60},{"city": "深圳", "temperature": 30}]

    #1，实例化转换器类
    transfer = DictVectorizer()

    #2, 调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)





if __name__ == '__main__':
    # datasets_demo()
    dict_demo()