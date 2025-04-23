import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer as SparkTokenizer, StopWordsRemover
from pyspark.ml.feature import Word2Vec as SparkWord2Vec
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# NLTK设置
import os
os.environ['NLTK_DATA'] = 'C:/Users/cdf/nltk_data'
try:
    nltk.data.path.append('C:/Users/cdf/nltk_data')
    nltk.download('punkt', download_dir='C:/Users/cdf/nltk_data')
    nltk.download('stopwords', download_dir='C:/Users/cdf/nltk_data')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"下载NLTK数据时出错: {e}")
    exit(1)

# 数据加载和预处理函数
def load_and_preprocess_data():
    df = pd.read_json(r"C:/Users/cdf/PycharmProjects/SparkRdd/kaggle/archive/Sarcasm_Headlines_Dataset_v2.json", lines=True)
    df.drop('article_link', axis=1, inplace=True)
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        return text
    
    df['headline'] = df['headline'].apply(clean_text)
    
    stop_words = set(stopwords.words('english'))
    def process_text(text):
        tokens = word_tokenize(text)
        return [word for word in tokens if word not in stop_words]
    
    df['tokens'] = df['headline'].apply(process_text)
    return df

# GloVe嵌入加载函数
def load_glove_embeddings(glove_path):
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# 组合嵌入矩阵创建函数
def create_combined_matrix(w2v_model, glove_embeddings, word_index, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim * 2))
    for word, i in word_index.items():
        if word in w2v_model.wv and word in glove_embeddings:
            embedding_matrix[i] = np.concatenate(
                (w2v_model.wv[word], glove_embeddings[word])
            )
    return embedding_matrix
# Keras模型训练部分
def train_keras_model(df):
    print("========== Training Keras Model ==========")
    X = df['tokens']
    y = df['is_sarcastic']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Word2Vec训练
    w2v_model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)
    glove_embeddings = load_glove_embeddings(r'C:\Users\cdf\PycharmProjects\SparkRdd\kaggle\glove.twitter.27B.100d.txt\glove.twitter.27B.100d.txt')
    
    # 文本序列化
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding='post')
    
    # 构建模型
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=200, 
                 input_length=100, weights=[create_combined_matrix(w2v_model, glove_embeddings, tokenizer.word_index, 100)],
                 trainable=False),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    history = model.fit(X_train_pad, y_train, batch_size=64, epochs=10, validation_data=(X_test_pad, y_test))
    
    # 模型评估
    print("\n========== Keras Model Evaluation ==========")
    y_pred = (model.predict(X_test_pad) > 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Non-Sarcastic', 'Sarcastic']))
    
    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Keras Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Spark模型训练部分
def train_spark_model(df):
    print("\n========== Training Spark Model ==========")
    spark = SparkSession.builder \
        .appName("SarcasmDetection") \
        .getOrCreate()
    
    spark_df = spark.createDataFrame(df)
    
    # 平衡数据集
    majority_class = spark_df.filter(spark_df["is_sarcastic"] == 0)
    minority_class = spark_df.filter(spark_df["is_sarcastic"] == 1)
    upsampled_minority = minority_class.sample(withReplacement=True, fraction=2.0, seed=42)
    balanced_df = majority_class.union(upsampled_minority)
    
    # 构建流水线
    pipeline = Pipeline(stages=[
        SparkTokenizer(inputCol="headline", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered_words"),
        SparkWord2Vec(vectorSize=200, minCount=1, windowSize=8, inputCol="filtered_words", outputCol="features"),
        MultilayerPerceptronClassifier(
            layers=[200, 128, 64, 32, 2],
            labelCol="is_sarcastic",
            maxIter=20,
            blockSize=128,
            seed=42
        )
    ])
    
    # 拆分数据集并训练
    train_df, test_df = balanced_df.randomSplit([0.8, 0.2], seed=42)
    pipeline_model = pipeline.fit(train_df)
    predictions = pipeline_model.transform(test_df)
    
    # 模型评估
    evaluator = MulticlassClassificationEvaluator(
        labelCol="is_sarcastic", 
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"Spark Model Test Accuracy: {accuracy:.4f}")
    
    # 显示预测结果示例
    print("\nSpark Model Prediction Samples:")
    predictions.select("prediction", "is_sarcastic").show(50)
    spark.stop()
# 主程序
if __name__ == "__main__":
    df = load_and_preprocess_data()
    #train_keras_model(df)
    train_spark_model(df)