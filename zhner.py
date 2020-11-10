
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from processData.data import pad_sequences, batch_yield_data


class ZHNer(object):
    def __init__(self, args, tag2label, vocab, config):
        self.batch_size = args["batch_size"]
        self.epoch_num = args["epoch"]
        self.hidden_dim = args["hidden_dim"]
        self.CRF = bool(args["crf"])  # CRF使能
        self.update_embedding = bool(args["update_embedding"])
        self.dropout_keep_prob = args["dropout"]
        self.optimizer = args["optimizer"]
        self.lr = args["lr"]
        self.clip_grad = args["clip"]
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab  # 字典：字->id
        self.shuffle = False
        self.unk = args["unk"]
        self.config = config

    def demo_one(self, sess, sent):
        """
        直接预测，并对预测出来的序列进行转码
        :param sess:
        :param sent:
        :return:
        """
        label_list = []
        for seqs in batch_yield_data(sent, self.batch_size, self.vocab, unk=self.unk):
            label_list_, _ = self.predict_one_batch(sess, seqs, demo=False)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        tag = [[label2tag[label] for label in text_label] for text_label in label_list]
        return tag

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """
        为预定义的每个placeholder绑定数据，将当前batch的输入数据组织成list
        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list, max_len = pad_sequences(seqs, pad_mark=0)
        # 对输入字id的list做zero padding到此batch最大句子长度，但是保留每个句子的实际长度

        feed_dict = {"word_ids:0": word_ids,
                     "sequence_lengths:0": seq_len_list,
                     "max_length:0": max_len,
                     "batch_size:0": len(seq_len_list)}
        if labels is not None: # dev / eval时可能为空
            labels_, _, _ = pad_sequences(labels, pad_mark=0) # 同上， 对label id 的list作 padding
            feed_dict["labels:0"] = labels_
        if lr is not None:
            feed_dict["lr_pl:0"] = lr
        if dropout is not None:
            feed_dict["dropout:0"] = dropout

        return feed_dict, seq_len_list # 在dev/eval时候要使用句子长度，参与计算CRF

    def predict_one_batch(self, sess, seqs, demo=False):
        """
        通过session调用网络计算到FC层为止，并取出模型目前的CRF转移矩阵
         使用转移矩阵对发射概率矩阵（全连接的结果）进行解码，返回评分最高的序列和序列的评分
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        """ 通过session调用网络计算到FC层为止，并取出模型目前的CRF转移矩阵 """
        logits, transition_params = sess.run([tf.get_default_graph().get_tensor_by_name("logits:0"),
                                              tf.get_default_graph().get_tensor_by_name("transitions:0")],
                                             feed_dict=feed_dict)
        if demo:
            print("发射矩阵如下", logits, logits.shape)
            print("转移概率矩阵如下", transition_params, transition_params.shape)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            if not seq_len:
                continue
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)  # 使用转移矩阵对发射概率矩阵（全连接的结果）进行解码，返回评分最高的序列
            label_list.append(viterbi_seq)
        return label_list, seq_len_list
