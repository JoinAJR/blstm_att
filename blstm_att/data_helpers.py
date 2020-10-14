import numpy as np
import pandas as pd
# import nltk
import re
import jieba
import utils

import utils_for_clinic
from configure import FLAGS


# 加载句子，关系，etype（句子和etype分开返回）
def load_data_and_labels_and_etypes(path,epath):
    data = []
    lines = [line.strip() for line in open(path, "r", encoding="utf-8")]
    etype_lines = [line.strip() for line in open(epath, "r", encoding="utf-8")]
    max_sentence_length = 0

    # jieba.load_userdict("C:/ProgramData/Anaconda3/Lib/site-packages/jieba/user-dict.txt")
    for idx in range(0, len(lines), 3):
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        etypes = etype_lines[idx].split("\t")[1]

        sentence = lines[idx].split("\t")[1][0:]
        sentence = sentence.replace('<e1>', ' e11 ')
        sentence = sentence.replace('</e1>', ' e12 ')
        sentence = sentence.replace('<e2>', ' e21 ')
        sentence = sentence.replace('</e2>', ' e22 ')

        tokens = jieba.cut(sentence, cut_all=False)
        tokens_list = []
        index = 0

        for i in tokens:
            tokens_list.append(i)

        if max_sentence_length < len(tokens_list):
            max_sentence_length = len(tokens_list)

        sentence = " ".join(tokens_list)

        data.append([id, sentence, relation, etypes])

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation", "etypes"])
    df['label'] = [utils_for_clinic.class2label_2[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()

    # Label Data
    y = df['label']

    #etype Data
    x_etypes = df['etypes'].tolist()

    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0]
    # 1  => [0 1 0 0]
    # ...
    # 3 => [0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, labels, x_etypes


# 加载句子，关系，etype（etype拼接到句子后）
def load_dataWithetypes_and_labels(path,epath):
    data = []
    lines = [line.strip() for line in open(path, "r", encoding="utf-8")]
    etype_lines = [line.strip() for line in open(epath, "r", encoding="utf-8")]
    max_sentence_length = 0

    # jieba.load_userdict("C:/ProgramData/Anaconda3/Lib/site-packages/jieba/user-dict.txt")
    for idx in range(0, len(lines), 3):
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        etypes = etype_lines[idx].split("\t")[1]

        sentence = lines[idx].split("\t")[1][0:]
        sentence = sentence.replace('<e1>', ' e11 ')
        sentence = sentence.replace('</e1>', ' e12 ')
        sentence = sentence.replace('<e2>', ' e21 ')
        sentence = sentence.replace('</e2>', ' e22 ')

        tokens = jieba.cut(sentence, cut_all=False)
        tokens_list = []
        index = 0

        for i in tokens:
            tokens_list.append(i)

        if max_sentence_length < len(tokens_list):
            max_sentence_length = len(tokens_list)

        sentence = " ".join(tokens_list)

        # 拼接etype到sentence
        sentence+=" "+etypes

        data.append([id, sentence, relation])

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    df['label'] = [utils_for_clinic.class2label_2[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()

    # Label Data
    y = df['label']

    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0]
    # 1  => [0 1 0 0]
    # ...
    # 3 => [0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, labels


# 加载句子，时序关系labels
def load_data_and_labels(path):
    data = []
    lines = [line.strip() for line in open(path, "r", encoding="utf-8")]
    max_sentence_length = 0
    desc1 = []
    desc2 = []
    # jieba.load_userdict("C:/ProgramData/Anaconda3/Lib/site-packages/jieba/user-dict.txt")
    for idx in range(0, len(lines), 3):
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][0:]
        sentence = sentence.replace('<e1>', ' e11 ')
        sentence = sentence.replace('</e1>', ' e12 ')
        sentence = sentence.replace('<e2>', ' e21 ')
        sentence = sentence.replace('</e2>', ' e22 ')

        # e1_Entity = sentence.split("e11 ")[1].split(" e12")[0]
        # e2_Entity = sentence.split("e21 ")[1].split(" e22")[0]


        # kb_desc = {
        #     "治疗": "干预或改变特定健康状态的过程",
        #     "手术": "医生用医疗器械对病人身体进行的切除、缝合等治疗",
        #     "住院": "病人住进医院接受治疗或观察",
        #     "出院": "在医院住院的病人结束住院，离开医院"
        # }
        #
        # kb_type = {
        #     "治疗": "治疗",
        #     "手术": "治疗",
        #     "住院": "治疗",
        # }

        # # 把实体的描述信息拿到
        # e1_desc = kb_desc.get(e1_Entity, e1_Entity);
        # e2_desc = kb_desc.get(e2_Entity, e2_Entity);
        # # e1_desc = e1_Entity
        # # e2_desc = e2_Entity
        #
        # e1_type = kb_type.get(e1_Entity, "");
        # e2_type = kb_type.get(e2_Entity, "");
        #
        # e1_desc_tokens = jieba.cut(e1_desc, cut_all=False)
        # e2_desc_tokens = jieba.cut(e2_desc, cut_all=False)
        # desc1_list = []
        # desc2_list = []

        # for i in e1_desc_tokens:
        #     desc1_list.append(i)
        # for i in e2_desc_tokens:
        #     desc2_list.append(i)
        # # desc1.append(desc1_list)
        # # desc2.append(desc2_list)
        # desc1.append(" ".join(desc1_list))
        # desc2.append(" ".join(desc2_list))
        # print(desc1_list, desc2_list)

        tokens = jieba.cut(sentence, cut_all=False)
        tokens_list = []
        index = 0

        for i in tokens:
            tokens_list.append(i)
            # if i == 'e11':
            #     e1_index = index+2
            # if i == 'e21':
            #     e2_index = index+2
            # index = index + 1

        # type_list = []
        # type_list.append(e1_type)
        # type_list.append(e2_type)
        # type_index = []
        # type_index.append(-1 if e1_type == "" else e1_index)
        # type_index.append(-1 if e2_type == "" else e2_index)
        
        if max_sentence_length < len(tokens_list):
            max_sentence_length = len(tokens_list)

        # tokens_list.append(e1_desc)
        # tokens_list.append(e2_desc)

        sentence = " ".join(tokens_list)
        # type_list = " ".join(type_list)

        data.append([id, sentence, relation])
        # data.append([id, sentence, relation, type_list,type_index])

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    # df = pd.DataFrame(data=data, columns=["id", "sentence", "relation", "word_type", "type_index"])
    #---------- df['label'] = [utils.class2label_2[r] for r in df['relation']]
    df['label'] = [utils_for_clinic.class2label_2[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()

    # Label Data
    y = df['label']

    # # type data
    # wType = df['word_type'].tolist()
    # type_index = df['type_index'].tolist()

    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0]
    # 1  => [0 1 0 0]
    # ...
    # 3 => [0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    # return x_text, labels, desc1, desc2, wType, type_index
    return x_text, labels, desc1, desc2


def load_data(path):
    data = []
    lines = [line.strip() for line in open(path, "r", encoding="utf-8")]
    max_sentence_length = 0
    # jieba.load_userdict("C:/ProgramData/Anaconda3/Lib/site-packages/jieba/user-dict.txt")
    for idx in range(0, len(lines), 3):
        id = lines[idx].split("\t")[0]
        # relation = lines[idx + 1]
        # print(idx)

        sentence = lines[idx].split("\t")[1][0:]
        sentence = sentence.replace('<e1>', ' e11 ')
        sentence = sentence.replace('</e1>', ' e12 ')
        sentence = sentence.replace('<e2>', ' e21 ')
        sentence = sentence.replace('</e2>', ' e22 ')

        tokens = jieba.cut(sentence, cut_all=False)
        tokens_list = []
        for i in tokens:
            tokens_list.append(i)

        if max_sentence_length < len(tokens_list):
            max_sentence_length = len(tokens_list)

        sentence = " ".join(tokens_list)

        data.append([id, sentence])

    # print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence"])

    # Text Data
    x_text = df['sentence'].tolist()


    return x_text


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    trainFile = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    testFile = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

    load_data_and_labels(testFile)
