import tensorflow as tf
from bert import modeling
import os
import create_input
import tokenization
import numpy as np
import optimization



# 这里是下载下来的bert配置文件

bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")
vocab_file="chinese_L-12_H-768_A-12/vocab.txt"
batch_size=20
num_labels=2# 类别数量，我的例子是个二分类
is_training=True
max_seq_length=128
iter_num=1000
lr=0.00005
if max_seq_length > bert_config.max_position_embeddings: # 模型有个最大的输入长度 512
    raise ValueError("超出模型最大长度")

# 加载数据集合
with open("data/text.txt","r",encoding="utf-8") as reader:
    data=reader.read().splitlines()



texts=[]
labels=[]
for line in data:
    line=line.split("\t")
    if len(line) ==2 and int(line[1])<2:# 这里演示一个二分类问题，但训练样本并没有认真处理过，所以去掉label大于1的。
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
labels=np.asarray(labels,dtype=np.int32)
#  创建bert的输入
input_ids=tf.placeholder (shape=[batch_size,max_seq_length],dtype=tf.int32,name="input_ids")
input_mask=tf.placeholder (shape=[batch_size,max_seq_length],dtype=tf.int32,name="input_mask")
segment_ids=tf.placeholder (shape=[batch_size,max_seq_length],dtype=tf.int32,name="segment_ids")
###
input_labels=tf.placeholder (shape=batch_size,dtype=tf.int32,name="input_ids")
# 创建bert模型
model = modeling.BertModel(
    config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
)

output_layer = model.get_sequence_output()# 这个获取每个token的output 输入数据[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
output_layer = model.get_pooled_output() # 这个获取句子的output
hidden_size = output_layer.shape[-1].value #获取输出的维度


# 后面就简单了，就是一个全连接，效果和下面的是一样的
# logits = tf.layers.dense(output_layer, 2)
# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_labels, name="soft_loss")
# loss = tf.reduce_mean(loss, name="loss")
# predict = tf.argmax(tf.nn.softmax(logits), axis=1, name="predictions")
# acc = tf.reduce_mean(tf.cast(tf.equal(input_labels, tf.cast(predict, dtype=tf.int32)), "float"), name="accuracy")


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
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(input_labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    predict = tf.argmax(tf.nn.softmax(logits), axis=1, name="predictions")
    acc = tf.reduce_mean(tf.cast(tf.equal(input_labels, tf.cast(predict, dtype=tf.int32)), "float"), name="accuracy")




# self.train_op = optimization.create_optimizer(
#     self.total_loss, FLAGS.learning_rate,num_train_steps, num_warmup_steps, False)

# 上面是官方提供的 optimization，下面是我直接把optimization.create_optimizer里的代码拷贝过来 做调试的。
#  官方提供的 optimization 主要是学习速率可以动态调整，如下面简图，学习速率由小到大，峰值就是设置的lr,然后在慢慢变小，
# 整个学习速率，呈现三角形
#
#                 -
#               -      -
#             -          -
#           -                 -
#

num_train_steps = int(  # 这个是总的迭代次数，训练完一轮的次数*总的次数
    len(texts) / batch_size * iter_num)
num_warmup_steps = int(num_train_steps * 0.1)  # 总的迭代次数 * 0.1 ,这里的0.1 是官方给出的，我直接写过来了

global_step = tf.train.get_or_create_global_step()
learning_rate = tf.constant(value=lr, shape=[], dtype=tf.float32)
# Implements linear decay of the learning rate.
learning_rate = tf.train.polynomial_decay(
    learning_rate,
    global_step,
    num_train_steps,
    end_learning_rate=0.0,
    power=1.0,
    cycle=False)
# Implements linear warmup. I.e., if global_step < num_warmup_steps, the
# learning rate will be `global_step/num_warmup_steps * init_lr`.
if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
optimizer = optimization.AdamWeightDecayOptimizer(
    learning_rate=learning_rate,
    weight_decay_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6,
    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
tvars = tf.trainable_variables()
grads = tf.gradients(loss, tvars)
# This is how the model was pre-trained.
(grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
train_op = optimizer.apply_gradients(  # 在这个方法内部 可以调整是否需要训练 BERT 参数
    zip(grads, tvars), global_step=global_step)
new_global_step = global_step + 1
train_op = tf.group(train_op, [global_step.assign(new_global_step)])

# train_op = tf.train.AdamOptimizer(lr).minimize(loss) # 这个是传统的optimization


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
lr_list=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iter_num):
        shuffIndex = np.random.permutation(np.arange(len(texts)))[:batch_size]
        batch_labels = labels[shuffIndex]
        batch_input_idsList=input_idsList[shuffIndex]
        batch_input_masksList=input_masksList[shuffIndex]
        batch_segment_idsList=segment_idsList[shuffIndex]
        l,a,_=sess.run([loss,acc,train_op],feed_dict={
            input_ids:batch_input_idsList,input_mask:batch_input_masksList,
            segment_ids:batch_segment_idsList,input_labels:batch_labels
        })

        print("准确率:{},损失函数:{}".format(a,l))

