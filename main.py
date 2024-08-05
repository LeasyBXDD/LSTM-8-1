import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Attention, Concatenate, TimeDistributed, Lambda

# 定义输入特征数和类别数
num_features = 30
num_classes = 10

# 定义输入层
sequence_input = Input(shape=(None, num_features), dtype='float32')

# 双向LSTM层
lstm_out = Bidirectional(LSTM(64, return_sequences=True))(sequence_input)

# 注意力机制
attention = Attention()([lstm_out, lstm_out])
context_vector = Concatenate(axis=-1)([lstm_out, attention])

# 全连接层 + Softmax 层
# 使用 TimeDistributed 对每个时间步进行预测
dense_output = TimeDistributed(Dense(num_classes))(context_vector)
output = tf.keras.layers.Softmax()(dense_output)

# 获取 t+1 时间步的输出
final_output = Lambda(lambda x: x[:, -1, :])(output)

# 定义模型
model = Model(inputs=sequence_input, outputs=final_output)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 输出模型结构
# model.summary()

# 测试模型

# 生成测试数据
batch_size = 5
sequence_length = 15  # 假设每个序列的长度为15
X_test = np.random.random((batch_size, sequence_length, num_features))

# print("Test data shape:", X_test.shape)

# 使用模型进行预测
predictions = model.predict(X_test)

# 打印预测结果
# print("Predictions shape:", predictions.shape)
print("活动概率分布 APD:")
print(predictions)  # 活动概率分布 APD
