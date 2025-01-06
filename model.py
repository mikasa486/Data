import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# 自定义函数，用于获取结巴分词默认过滤的部分停用词（简易版，仅作示意）
def get_default_stop_words():
    # 选取一段有代表性的文本示例，用于获取结巴分词时会过滤的词（实际中可多选取几段不同文本）
    sample_text = "的，地，了，得，景色，景区"
    words = jieba.cut(sample_text)
    filtered_words = []
    for word in words:
        if word == "":  # 简单假设分词结果为空字符串表示是被过滤的停用词（实际可能更复杂）
            filtered_words.append(word)
    return set(filtered_words)


# 数据导入
# 替换 'your_path/your_travel_reviews.csv' 为实际存放数据的文件路径
data = pd.read_csv('result_四川.csv', encoding='gbk')

# 获取结巴默认的部分停用词
stop_words = get_default_stop_words()


# 文本预处理函数
def preprocess_text(text):
    # 使用结巴分词对中文进行分词
    words = jieba.cut(text)
    # 去除停用词
    filtered_words = [word for word in words if word not in stop_words]
    # 将处理后的词重新组合为文本（以空格分隔单词）
    return " ".join(filtered_words)


# 对所有评论进行预处理
data['processed_text'] = data['text'].apply(preprocess_text)

# 划分训练集和测试集
# 特征（处理后的文本）
X = data['processed_text']
# 假设你的CSV文件中有一个名为 'sentiment' 的列用于存放情感标签，若实际列名不同需修改此处
y = data['sentiment']
# 划分训练集和测试集，可根据实际数据量等情况调整 test_size 参数
# random_state 参数用于保证每次划分结果的一致性，方便对比不同参数设置下模型的性能
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 文本向量化
# 创建词袋模型向量器，可调整 max_features 参数
vectorizer = CountVectorizer(max_features=5000)
# 拟合训练集数据并转换为向量
X_train_vectors = vectorizer.fit_transform(X_train)
# 转换测试集数据为向量（用训练集拟合的词袋模型来转换）
X_test_vectors = vectorizer.transform(X_test)

# 模型选择与训练（这里使用朴素贝叶斯分类器）
# 可调整 alpha 参数
model = MultinomialNB(alpha=1.0)
# 用训练集向量和对应的标签进行训练
model.fit(X_train_vectors, y_train)

# 模型评估
# 在测试集上进行预测
y_pred = model.predict(X_test_vectors)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
# 查看详细的分类报告（包含精确率、召回率、F1值等指标）
print(classification_report(y_test, y_pred))
