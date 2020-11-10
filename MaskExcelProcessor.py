#-*-encoding:utf8-*-#

import os
import pandas as pd
import tensorflow as tf
import warnings
from zhner import ZHNer
from processData.data import read_dictionary, tag2label
from utils import get_multiple_entity
warnings.filterwarnings('ignore')
import re

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
cwd = os.getcwd()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 按需分配GPU
config.gpu_options.per_process_gpu_memory_fraction = 1.0  # 分配固定大小最多占显存的0.2 need ~700MB GPU memory

"""
生产任务：读取指定文件夹下的PDF，进行生产环境处理
"""
class MaskingProcessor():
    ner_model = None
    search_key = {
        "phone": "电话|手机|号码",
        "id": "身份证|证件",
        "address": "地址|住"
    }

    search_re = {
        "phone": ["1[3-9][0-9嗯啊哦额呃]{3,}", "[3-8][0-9嗯啊哦额呃]{4,}", "0[0-9]{3}[3-8][0-9嗯啊哦额呃]{4,12}"],
        "id": ["[1-9][0-9嗯啊哦额呃]{5,20}"],
        "address": [".{1,2}[省市镇村路][0-9]{1,5}号楼?", ".{1,2}[省市镇村]", ".{1,2}街道"]
    }

    def __init__(self):
        self.model_path = os.path.join(cwd, 'model_save/ner/checkpoints/')
        self.creat_model()
        for key in self.search_re:
            self.search_re[key] = [re.compile(regular)
                                   for regular in self.search_re[key]]

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
    def run(self, file_name):
        # file_name = "客服话务原始文本及打标.xlsx"
        mask_column_key = "mask文本"

        df = pd.read_excel(file_name)
        columns = list(df.columns) + [mask_column_key]
        columns[-2], columns[-1] = columns[-1], columns[-2]  # 交换最后两列

        if '文本详情' not in columns:
            raise AttributeError("excel文件找不到名为 '文本详情' 的列，请检查文件是否正确")
        texts = df['文本详情']
        cnt = 0
        data_dict = {k: [] for k in columns}

        texts_entity_dict = self.get_texts_ner_res(texts)
        total = len(texts)
        print(data_dict)
        for text in texts:
            to_be_mask = set()
            idx = text.find("话务")
            for key in self.search_key:
                search_res = re.search(self.search_key[key], text)
                if search_res is None: continue
                if idx != -1:
                    #print(idx, text[idx:])
                    for rel in self.search_re[key]:
                        to_be_mask = to_be_mask.union(iter_search(text, rel, idx))

            ## 取NER
            entity_dict = texts_entity_dict[cnt]
            for key in entity_dict:
                to_be_mask = to_be_mask.union(set(entity_dict[key]))

            sort_idx = sorted(list(to_be_mask), key=lambda x: (x[0], -x[1]))
            right = -1
            final = []
            for pair in sort_idx:
                if pair[1] <= right: continue
                if pair[0] <= right:  #
                    final[-1] += text[right: pair[1]]
                    right = pair[1]
                else:
                    left = max(pair[0], right)
                    right = pair[1]
                    final.append(text[left: right])

            for key in df.columns:
                data_dict[key].append(df[key][cnt])
            data_dict[mask_column_key].append(",".join(final))
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

"""脚本的命令行输入提示"""
def printUsage():
    raise AssertionError("使用说明: \npython MaskExcelProcessor.py -f 要处理的excel文件名.xlsx")

if __name__ == "__main__":
    import getopt, sys

    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:", ["file="])
    except getopt.GetoptError:
        # 参数错误
        printUsage()
        sys.exit(-1)

    target_file = ""
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            target_file = arg
    if target_file == "":
        printUsage()

    processor = MaskingProcessor()
    processor.run(target_file)


