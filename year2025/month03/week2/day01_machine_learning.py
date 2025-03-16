from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import matplotlib.pyplot as plt



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


def count_demo():
    """
    文本特征抽取
    :return:
    """
    data = ["life is short, i like like python","life is too lang, i dislike python"]
    transfer = CountVectorizer(stop_words=["is","too"])
    date_new = transfer.fit_transform(data)
    # print("data_new: \n", date_new)
    print("data_new: \n", date_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())


def cut_word(text):
    """
    进行中文分词
    :param text:
    :return:
    """
    text = " ".join(list(jieba.cut(text)))
    return text


def count_chinese_demo():
    """
    中文文本特征提取，自动分词
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年前发出的，这样当我们看到宇宙时，我们是在看它的过去",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for i in data:
        data_new.append(cut_word(i))
    print("data_new: \n",data_new)

    transfer = CountVectorizer(stop_words=["一种","所以"])
    data_final = transfer.fit_transform(data_new)
    print("data_final: \n", data_final)
    print("特征名字：\n", transfer.get_feature_names_out())


def tfidf_demo():
    """
    用TF-IDF的方法进行文本特征抽取
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年前发出的，这样当我们看到宇宙时，我们是在看它的过去",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for i in data:
        data_new.append(cut_word(i))
    transfer = TfidfVectorizer()

    data_final = transfer.fit_transform(data_new)
    print("data_final: \n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())


def minmax_demo():
    """
    归一化
    :return:
    """
    data = pd.read_csv("dating.csv", sep='\t')
    data = data.iloc[:, :3]
    print("data: \n", data)

    # 可以选择不同的特征值范围
    # transfer = MinMaxScaler()
    transfer = MinMaxScaler()
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new)



def stand_demo():
    """
    标准化
    :return:
    """
    data = pd.read_csv("dating.csv", sep='\t')
    data = data.iloc[:, :3]
    print("data: \n",data)

    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new)



def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    data = pd.read_csv("factor_returns.csv")
    data = data.iloc[:, 1:-2]
    print("data: \n",data)

    transfer = VarianceThreshold(threshold=10)
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new)
    print("data_new: \n", data_new.shape)

    r1 = pearsonr(data['pe_ratio'], data['pb_ratio'])
    print("相关系数：\n", r1)

    r2 = pearsonr(data['revenue'], data['total_expense'])
    print("revenue和total_expense相关性：\n", r2)

    plt.scatter(data['revenue'], data['total_expense'])
    plt.show()



if __name__ == '__main__':
    # datasets_demo()
    # dict_demo()
    # count_demo()
    # str = "我爱北京天安门"
    # cut_word(str)
    # count_chinese_demo()
    # tfidf_demo()
    # minmax_demo()
    # stand_demo()
    variance_demo()