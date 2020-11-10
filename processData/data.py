#-*-encoding:utf8-*-#
import pickle, os
import numpy as np

tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }

def original2NERinput(content_dict):
    """
    :param content_dict: { "pageContent": {1: [page1's Contents list], 2:[page2's contents list]...}}
    :param tag_schema:
    :return: pageIdxRange: NER_input中每行数据属于pdf的第几页 list(range(1,x), range(x,x2)...)
              NER_input: 每句pdf原文的逐字拆分列表
    """
    line_cnt = 0
    pageIdxRange = {}  # 记录每页的句子idx的范围，之后用来整理NER的结果
    NER_input = []
    for page in sorted(content_dict.keys()):
        pageBegin = line_cnt
        for sen_struct in content_dict[page]:
            content = sen_struct["origin"]
            line_cnt += 1
            char_list = list(content[:400])
            NER_input.append(char_list)
        pageEnd = line_cnt
        pageIdxRange[page] = (pageBegin, pageEnd)
    return pageIdxRange, NER_input


"""  2019-5-22:为了能够和带有单个数字/字母的pretrain char embedding兼容，调整代码结构，在没有预训练词向量的时候保留原功能"""
def sentence2id(sent, word2id, unk='<UNK>'):
    """
    字转id，其中数字一律以数字标签处理，英文一律以英文标签处理，
    :param sent:
    :param word2id:
    :return: 字的id list
    """
    sentence_id = []
    for word in sent:
        if unk == '<UNK>':
            if word not in word2id:
                if word.isdigit():
                    word = '<NUM>'
                elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                    word = '<ENG>'
                else:
                    word = '<UNK>'
        else:
            if word not in word2id:
                word = unk
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    #print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    u"""
    随机初始化一个均匀分布[-0.25, 0.25]，shape为(字典长*embedding_dim)
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences)) # 根据当前batch的字id列表的长度，计算当前batch的最大句子长度
    seq_list, seq_len_list = [], []
    ## 对输入字id的list做zero padding，但是保留每个句子的实际长度 （只是为了创建一个batch的张量不失败？）
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list, max_len

def batch_yield_data(data, batch_size, vocab, unk='<UNK>'):
    """
    处理测试数据为batch数据，包括：字映射到id
    数字与英文均分别处理为统一标识符
    :param data:
    :param batch_size:
    :param vocab:
    :param unk:
    :return:
    """

    seqs = []
    for sent_ in data:
        if len(sent_) > 3000:
            sent_ = sent_[:3000]
        sent_ = sentence2id(sent_, vocab, unk)  # 句子里的每个字的id构成的list

        if len(seqs) == batch_size:
            yield seqs  # 积累的数据达到batch_size，返回当前积累的数据，并清空当前batch，下次继续
            seqs = []
        seqs.append(sent_)

    if len(seqs) != 0:
        # seqs.extend([[]] * (batch_size-len(seqs)))
        yield seqs

