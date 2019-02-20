import tensorflow as tf
from bert import modeling
import os
import create_input
import tokenization
import numpy as np

# 这里是下载下来的bert配置文件
bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")
vocab_file="chinese_L-12_H-768_A-12/vocab.txt"
num_labels=20
is_training=True
max_seq_length=80
iter_num=1000
lr=0.01
if max_seq_length > bert_config.max_position_embeddings: # 模型有个最大的输入长度 512
    raise ValueError("超出模型最大长度")

# 加载数据集合
with open("data/text.txt","r",encoding="utf-8") as reader:
    data=reader.read().splitlines()



texts=[]
labels=[]
for line in data:
    line=line.split("\t")
    if len(line) ==2:
        texts.append(line[0])
        labels.append(line[1])

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file) # token 处理器，主要作用就是 分字，将字转换成ID。vocab_file 字典文件路径
input_idsList=[]
input_masksList=[]
segment_idsList=[]
for t in texts:
    single_input_id, single_input_mask, single_segment_id=create_input.convert_single_example(max_seq_length,tokenizer,t)
    input_idsList.append(single_input_id)
    input_masksList.append(single_input_mask)
    segment_idsList.append(single_segment_id)



input_idsList=np.asarray(input_idsList,dtype=np.int32)
input_masksList=np.asarray(input_masksList,dtype=np.int32)
segment_idsList=np.asarray(segment_idsList,dtype=np.int32)
labels=np.asarray(labels,dtype=np.int32).reshape([-1,1])
#  创建bert的输入
input_ids=tf.placeholder (shape=[num_labels,max_seq_length],dtype=tf.int32,name="input_ids")
input_mask=tf.placeholder (shape=[num_labels,max_seq_length],dtype=tf.int32,name="input_mask")
segment_ids=tf.placeholder (shape=[num_labels,max_seq_length],dtype=tf.int32,name="segment_ids")
###
input_labels=tf.placeholder (shape=[num_labels,1],dtype=tf.int32,name="input_ids")
# 创建bert模型
model = modeling.BertModel(
    config=bert_config,
    is_training=True,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
)

output_layer = model.get_sequence_output()# 这个获取每个token的output 输入数据[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
output_layer = model.get_pooled_output() # 这个获取句子的output
hidden_size = output_layer.shape[-1].value #获取输出的维度

# 后面就简单了，就是一个全连接
output_weights = tf.get_variable(
    "output_weights", [num_labels, hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02))

output_bias = tf.get_variable(
    "output_bias", [num_labels], initializer=tf.zeros_initializer())
with tf.variable_scope("loss"):
    if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    # probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(input_labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    predictions = tf.reshape(tf.argmax(logits, axis=-1, output_type=tf.int32),shape=[-1,1],name="predictions")
    acc = tf.reduce_mean(tf.cast(tf.equal(input_labels, predictions), "float"),
                         name="accuracy")

train_op = tf.train.AdamOptimizer(lr).minimize(loss)


#bert模型参数初始化的地方
init_checkpoint = "chinese_L-12_H-768_A-12/bert_model.ckpt"
use_tpu = False
# 获取模型中所有的训练参数。
tvars = tf.trainable_variables()
# 加载BERT模型
(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                       init_checkpoint)

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

tf.logging.info("**** Trainable Variables ****")
# 打印加载模型的参数
for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iter_num):
        shuffIndex = np.random.permutation(np.arange(num_labels))
        batch_labels = labels[shuffIndex]
        batch_input_idsList=input_idsList[shuffIndex]
        batch_input_masksList=input_masksList[shuffIndex]
        batch_segment_idsList=segment_idsList[shuffIndex]
        l,a,_=sess.run([loss,acc,train_op],feed_dict={
            input_ids:batch_input_idsList,input_mask:batch_input_masksList,
            segment_ids:batch_segment_idsList,input_labels:batch_labels
        })
        print("准确率:{},损失函数:{}".format(a,l))