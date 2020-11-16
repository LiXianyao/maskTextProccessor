#-*-encoding:utf8-*-#

import os
import pandas as pd
import tensorflow as tf
import warnings
from zhner import ZHNer
import json
from processData.data import read_dictionary, tag2label
from utils import get_multiple_entity, str2bool
warnings.filterwarnings('ignore')
import re

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
cwd = os.getcwd()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 按需分配GPU
config.gpu_options.per_process_gpu_memory_fraction = 1.0  # 分配固定大小最多占显存的0.2 need ~700MB GPU memory

"""
生产任务：读取指定文件夹下的PDF，进行生产环境处理
"""
class MaskingProcessor():

    def __init__(self):
        self.model_path = os.path.join(cwd, 'model_save/ner/checkpoints/')
        self.creat_model()
        self.load_rules_from_file()


    def load_rules_from_file(self):
        with open("manualFeatures.json", "r") as rules_file:
            defined_rules = json.load(rules_file)
            self.remove_words = defined_rules["清除词"]
            self.search_key = {key: value["key"] for (key, value) in defined_rules["规则识别"].items()}
            self.search_re = {key: value["rules"] for (key, value) in defined_rules["规则识别"].items()}
            for key in self.search_re:
                self.search_re[key] = [re.compile(regular)
                                       for regular in self.search_re[key]]
            self.global_re = [re.compile(rules)
                                for rules in defined_rules["特定规则"]]

    def creat_model(self, config_file="ner.config"):
        from config import load_config_file
        task_dict, hyper_parameters_dict, extra_parameters_dict, = load_config_file(configFile=config_file)
        u""" 读取预处理的word2id文件（实际上是每个字分配一个id) """
        word2id = read_dictionary(os.path.join('embeddings', extra_parameters_dict["word2id"]))

        ## paths setting
        """ 处理对模型结果等文件的保存名字及路径, 以及logger的保存位置 """
        ckpt_file = tf.train.latest_checkpoint(self.model_path)
        self.model = ZHNer(hyper_parameters_dict, tag2label, word2id, config=config)
        self.saver = tf.train.import_meta_graph(ckpt_file + ".meta")
        self.sess = tf.Session(config=config)
        self.saver.restore(self.sess, ckpt_file)
        print("--------- INFO:checkpoint还原模型完成！,checkpoint = {}".format(self.model_path))

    """（调试用）加载一个现成的结果json文件导出到csv"""
    def run(self, file_name, keep_origin):
        # file_name = "客服话务原始文本及打标.xlsx"
        mask_column_key = "mask内容"
        masked_text_column_key = "处理后文本"
        text_column = '文本详情'

        df = pd.read_excel(file_name)
        if keep_origin:  # 保留原文的模式（调试用）
            columns = list(df.columns) + [mask_column_key, masked_text_column_key]
        else:
            columns = list(df.columns) + [masked_text_column_key]
            columns.remove(text_column)
        keep_columns = set(df.columns).intersection(columns)

        if text_column not in df.columns:
            raise AttributeError("excel文件找不到名为 '文本详情' 的列，请检查文件是否正确")
        texts = df[text_column]
        cnt = 0
        data_dict = {k: [] for k in columns}

        clear_texts, removed_words = self.remove_words_from(texts)
        trans_sents = self.trans_chinese2num(clear_texts)
        print("-----------INFO: 处理中，文本预处理完成，进入识别阶段（需要几分钟，请稍候） -------------------")

        texts_entity_dict = self.get_texts_ner_res(trans_sents)
        total = len(trans_sents)
        print(data_dict)
        for text in trans_sents:
            to_be_mask = set()
            idx = text.find("话务")
            for key in self.search_key:
                search_res = re.search(self.search_key[key], text)
                if search_res is None: continue
                if idx != -1:
                    #print(idx, text[idx:])
                    for rel in self.search_re[key]:
                        to_be_mask = to_be_mask.union(iter_search(text, rel, idx))

            for rel in self.global_re:
                to_be_mask = to_be_mask.union(iter_search(text, rel, idx))
            ## 取NER
            entity_dict = texts_entity_dict[cnt]
            for key in entity_dict:
                to_be_mask = to_be_mask.union(set(entity_dict[key]))

            sort_idx = sorted(list(to_be_mask), key=lambda x: (x[0], -x[1]))
            right = -1
            final = removed_words[cnt]
            for pair in sort_idx:
                if pair[1] <= right: continue
                if pair[0] <= right:  #
                    final[-1] += text[right: pair[1]]
                    right = pair[1]
                else:
                    left = max(pair[0], right)
                    right = pair[1]
                    final.append(text[left: right])

            for mask_word in final:
                text = text.replace(mask_word, "[XX]")

            for key in keep_columns:
                data_dict[key].append(df[key][cnt])
            data_dict[mask_column_key].append(",".join(final)) if origin else None
            data_dict[masked_text_column_key].append(text)
            #print(data_dict[mask_column_key])

            if cnt % (total // 30) == 0:
                print("-----------INFO: 处理中，已处理 {} 条数据，还剩 {}条，完成比例 {}% -------------------".
                      format(cnt, total - cnt, round(cnt * 100.0 / total, 2)))
            cnt += 1

        result_file_name = "".join(file_name.split(".")[:-1] + ["-MASK.xlsx"])
        writer = pd.ExcelWriter(result_file_name)
        df1 = pd.DataFrame(data=data_dict)
        df1.to_excel(writer, 'Sheet1', columns=columns, index=False)
        writer.save()
        print("---------FINISH：处理完毕，结果已保存到文件{} -------------------".format(result_file_name))

    def remove_words_from(self, sents):
        remove_records = []
        after_remove_sent = []
        for sid in range(len(sents)):
            sent = sents[sid]
            remove = []
            for word in self.remove_words:
                if sent.find(word) != -1:
                    remove.append(word)
                sent = sent.replace(word, "")
            after_remove_sent.append(sent)
            remove_records.append(remove)
        return after_remove_sent, remove_records

    def trans_chinese2num(self, sents):
        pattern = re.compile("[零一二三四五六七八九十]{2,}")
        for sid in range(len(sents)):
            sent = sents[sid]
            match_idxs = iter_search(sent, pattern, 0)
            for (left, right) in sorted(match_idxs, reverse=True):
                trans_str = chinese_to_num(sent[left: right], right - left, 0)
                sent = sent[:left] + trans_str + sent[right:]
            sents[sid] = sent
        return sents

    def get_ner_res(self, sent):
        sess = self.sess
        sent = list(sent.strip())
        demo_data = [sent]
        tag = self.model.demo_one(sess, demo_data)

        entity_dict = get_multiple_entity(tag[0], sent)
        return entity_dict

    def get_texts_ner_res(self, sents):
        sess = self.sess
        demo_data = [list(sent.strip()) for sent in sents]
        tag = self.model.demo_one(sess, demo_data)

        entity_dict = []
        for idx in range(len(tag)):
            entity_dict.append(get_multiple_entity(tag[idx], sents[idx]))
        return entity_dict



def iter_search(sent, reObj, idx):
    match_res = reObj.search(sent, idx)
    res = set()
    if match_res != None:
        match_span = match_res.span()
        res.add(match_span)
        res = res.union(iter_search(sent, reObj, match_span[-1]))
    return res

## 需要注意的是，转化的目标是号码、证件号等非实数数字，所以理论上没有百、千（口语念号码显然不会有）
ch2num = {"零": "0", "一": "1", "二":"2", "三":"3", "四":"4",
          "五":"5", "六":"6", "七":"7", "八":"8", "九":"9", "十": "1"}
def chinese_to_num(seq, len, depth):
    if len == 1:
        return ch2num[seq[0]]
    previous_str = chinese_to_num(seq[:-1], len - 1, depth + 1)
    now = ch2num[seq[-1]]
    if seq[-1] == "十":
        if depth == 0:
            now = "0"
        else:
            now = ""
    return previous_str + now

"""脚本的命令行输入提示"""
def printUsage():
    raise AssertionError("使用说明: \npython MaskExcelProcessor.py -f 要处理的excel文件名.xlsx")

if __name__ == "__main__":
    import getopt, sys

    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:o:", ["file=", "origin="])
    except getopt.GetoptError:
        # 参数错误
        printUsage()
        sys.exit(-1)

    target_file = ""
    origin = False
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            target_file = arg
        if opt in ("-o", "--origin"):
            origin = str2bool(arg)
    if target_file == "":
        printUsage()

    processor = MaskingProcessor()
    processor.run(target_file, origin)


