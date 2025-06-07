import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml import Pipeline
from pyspark import SparkContext
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gensim.models import Word2Vec
import os
os.environ['PYSPARK_PYTHON'] = "C:/Users/cdf/.conda/envs/tensorflow/python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = "C:/Users/cdf/.conda/envs/tensorflow/python.exe"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# 数据加载和预处理函数
def load_and_preprocess_with_spark(data_dir="archive"):
    spark = SparkSession.builder.appName("SarcasmDetection").getOrCreate()
    json_path = os.path.join(data_dir, "Sarcasm_Headlines_Dataset_v2.json")
    df = spark.read.json(json_path, multiLine=False)
    df = df.drop('article_link')

    tokenizer = Tokenizer(inputCol="headline", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    pipeline = Pipeline(stages=[tokenizer, remover])
    processed_df = pipeline.fit(df).transform(df)

    pandas_df = processed_df.select("headline", "filtered_words", "is_sarcastic").toPandas()
    pandas_df['text'] = pandas_df['filtered_words'].apply(lambda x: ' '.join(x))
    
    spark.stop()
    return pandas_df

# 加载 GloVe 词向量
def load_glove_embeddings(path, embedding_dim=100):
    embeddings_index = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Loaded {len(embeddings_index)} GloVe vectors.")
    return embeddings_index

# 训练 Word2Vec 模型
def train_word2vec(texts, vector_size=100, window=5, min_count=2):
    tokenized_texts = [text.split() for text in texts]
    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size,
                     window=window, min_count=min_count, workers=4, sg=1)
    print("Word2Vec model trained.")
    return model

# 构建组合 Embedding 矩阵（GloVe + Word2Vec）
def create_combined_embedding_matrix(word_index, glove_path, w2v_model, embedding_dim=100):
    glove_embeddings = load_glove_embeddings(glove_path, embedding_dim)

    num_words = min(10000, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim * 2))  # 双倍维度

    hits = 0
    misses = 0

    for word, i in word_index.items():
        if i >= num_words:
            continue

        w2v_vector = None
        if word in w2v_model.wv:
            w2v_vector = w2v_model.wv[word]

        glove_vector = None
        if word in glove_embeddings:
            glove_vector = glove_embeddings[word]

        if w2v_vector is not None and glove_vector is not None:
            embedding_matrix[i] = np.concatenate([w2v_vector, glove_vector])
            hits += 1
        else:
            vec = w2v_vector if w2v_vector is not None else glove_vector
            if vec is not None:
                embedding_matrix[i] = np.concatenate([vec, vec])  # 填充双份
                hits += 1
            else:
                misses += 1

    print(f"Found matches for {hits} words, missed {misses}.")
    return embedding_matrix

# 自定义 Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# 构建带 Attention 的 BiLSTM 模型
def build_attention_bilstm_model(embedding_matrix, maxlen=50, embedding_dim=200):
    inputs = Input(shape=(maxlen,))
    
    embedding = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    )(inputs)
    
    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(embedding)
    attention_out = AttentionLayer()(lstm_out)
    
    dense = Dense(64, activation='relu')(attention_out)
    dropout = Dropout(0.5)(dense)
    outputs = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def setup_spark_for_distributed_training():
    spark = SparkSession.builder \
        .appName("DistributedSarcasmTraining") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "1") \
        .config("spark.num.executors", "2") \
        .getOrCreate()
    sc = spark.sparkContext
    print(f"Using Python executable: {sc.pythonExec}")
    return spark, sc

def distributed_train(model, X_train_pad, y_train, embedding_matrix, maxlen, num_workers=2):
    try:
        spark, sc = setup_spark_for_distributed_training()
        data = np.hstack([X_train_pad, y_train.values.reshape(-1,1)])
        data = data.astype(np.float32)
        print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
        num_workers = min(num_workers, len(data) // 100 or 1)
        rdd_data = sc.parallelize(data, numSlices=num_workers)
        initial_weights = model.get_weights()
        broadcast_weights = sc.broadcast(initial_weights)
        broadcast_embedding = sc.broadcast(embedding_matrix)
        broadcast_maxlen = sc.broadcast(maxlen)
        
        def train_partition(partition):
            try:
                partition_data = list(partition)
                if not partition_data:
                    print("Empty partition, skipping...")
                    return []
                X_part = np.array([x[:-1] for x in partition_data], dtype=np.float32)
                y_part = np.array([x[-1] for x in partition_data], dtype=np.float32)
                if np.isnan(X_part).any() or np.isinf(X_part).any():
                    raise ValueError("X_part contains invalid values.")
                local_model = build_attention_bilstm_model(
                    broadcast_embedding.value, 
                    broadcast_maxlen.value
                )
                local_model.set_weights(broadcast_weights.value)
                local_model.fit(X_part, y_part, epochs=1, batch_size=64, verbose=0)
                return [local_model.get_weights()]
            except Exception as e:
                print(f"Worker异常: {str(e)}")
                return []
        
        weights_rdd = rdd_data.mapPartitions(train_partition)
        all_weights = weights_rdd.collect()
        valid_weights = [w for w in all_weights if w]
        if not valid_weights:
            raise ValueError("No valid weights collected.")
        avg_weights = [np.mean([w[i] for w in valid_weights], axis=0) for i in range(len(valid_weights[0]))]
        model.set_weights(avg_weights)
        return model
    finally:
        spark.stop()


# 主训练流程
def train_deep_learning_model(df):
    X = df['text']
    y = df['is_sarcastic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tokenizer = KerasTokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    maxlen = 50
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

    # 训练 Word2Vec
    w2v_model = train_word2vec(X_train, vector_size=100)

    # 创建组合词向量矩阵 (GloVe + Word2Vec)
    embedding_dim = 100
    # 修正 GloVe 路径（根据实际文件位置调整）
    glove_path = r"glove.twitter.27B.100d.txt\glove.twitter.27B.100d.txt"  
    embedding_matrix = create_combined_embedding_matrix(
        tokenizer.word_index, glove_path, w2v_model, embedding_dim
    )

    # 添加 OOV 向量
    oov_index = tokenizer.word_index.get('<OOV>', None)
    if oov_index and oov_index < embedding_matrix.shape[0]:
        embedding_matrix[oov_index] = np.mean(embedding_matrix[:oov_index], axis=0)

    # 构建模型
    model = build_attention_bilstm_model(embedding_matrix, maxlen, embedding_dim=200)
    print(model.summary())

    # 回调函数（早停）
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

    # 类别权重平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

 
    history = model.fit(X_train_pad, y_train, 
                        epochs=15, 
                        batch_size=64,
                        validation_data=(X_test_pad, y_test),
                        class_weight=class_weights_dict,
                        callbacks=[early_stop, reduce_lr])

    # 预测和评估
    y_proba = model.predict(X_test_pad)
    y_pred = (y_proba > 0.5).astype(int)

    print(classification_report(y_test, y_pred, target_names=["非讽刺", "讽刺"]))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba.ravel()):.4f}")

    # 绘制混淆矩阵和训练历史（代码不变）
    cm = confusion_matrix(y_test, y_pred)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('深度学习模型混淆矩阵')
    plt.xlabel('预测'), plt.ylabel('真实')
    plt.show()

    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_and_preprocess_with_spark()
    train_deep_learning_model(df)
