def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    # 将两个句子相加，如果长度大于max_length 就pop 直到小于 max_length
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def convert_single_example( max_seq_length,
                           tokenizer,text_a,text_b=None):
  tokens_a = tokenizer.tokenize(text_a)
  tokens_b = None
  if text_b:
    tokens_b = tokenizer.tokenize(text_b)# 这里主要是将中文分字
  if tokens_b:
    # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
    # 因为要为句子补上[CLS], [SEP], [SEP]
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 3
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
  # (a) 两个句子:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) 单个句子:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # 这里 "type_ids" 主要用于区分第一个第二个句子。
  # 第一个句子为0，第二个句子是1。在余训练的时候会添加到单词的的向量中，但这个不是必须的
  # 英文[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将中文转换成ids
  # 创建mask
  input_mask = [1] * len(input_ids)
  # 对于输入进行补0
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  return input_ids,input_mask,segment_ids