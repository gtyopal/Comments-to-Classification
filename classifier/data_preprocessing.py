# coding: utf-8
import sys
import os
import random
import re
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
import pandas as pd
from bs4 import BeautifulSoup
import re

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.encode('utf-8').strip().lower()

def strip_html(string):
    soup = BeautifulSoup(string, "html.parser")
    string = soup.get_text()
    r = re.compile(r'<[^>]+>', re.S)
    string = r.sub('', string)
    string = re.sub(r'&(nbsp;)', ' ', string)
    string = re.sub(r'<[^>]+', '', string)
    string = re.sub('\&lt[;]', ' ', string)
    string = re.sub('\&gt[;]', ' ', string)
    return string


def denoise_text(string):
    string = clean_str(string)
    string = strip_html(string)
    words = word_tokenize(string)
    string = " ".join(words)
    if not string.strip():
        string = " "
    return string


def load_data_from_disk(input_dataset_path, output_dataset_path):
    df = pd.read_csv(input_dataset_path, sep=",", encoding='latin-1', low_memory=False, error_bad_lines = False, header= None, usecols=[0, 1, 2],skiprows=1)
    df.rename(columns={0: 'row_id', 1: "message", 2: "label" }, inplace=True)
    df = df.dropna()
    df["text"] = df["message"]
    df = df.drop_duplicates(["text"])
    df['text'] = df['text'].apply(denoise_text)
    df = df[df["text"] != " "]
    print('number of sentences:', df.shape[0])
    nb_classes = len(list(set(df['label'].tolist())))
    print("number of tags is", nb_classes)
    df.to_csv(output_dataset_path, index=False)
    return df, nb_classes


# Word dictionary prepare and data split
def generate_words(df_dataset):
    """ generate words dict """
    words_dict = defaultdict(int)
    for item in df_dataset["text"]:
        if isinstance(item, float) or isinstance(item, int):
            continue
        for word in item.split():
            word = word.lower()
            if re.findall(r'^[a-zA-Z0-9\.\?\!\`\"\;\:\.\,\@\#\$\(\)\-\_\+\=\^\%\&\*]+$', word):
                words_dict[word] += 1
    count_sort = sorted(words_dict.items(), key=lambda e:-e[1])
    word2id = {'<pad>': 0}
    idx = 1
    for w in count_sort:
        if w[1] > 1:
            word2id[w[0]] = idx
            idx += 1
    return word2id

# save words and id
def save_words(word2id, words_path):
    with open(words_path, "w") as fw:
        for w in word2id:
            fw.write(w + "\t" + str(word2id[w]) + "\n")
    return 0

# load words and id
def load_words(words_path):
    word2id = {}
    with open(words_path, "r") as fr:
        for line in fr.readlines():
            line = line.strip().split("\t")
            word2id[line[0]] = int(line[1])
    return word2id


def generate_category2id(df_dataset):
    category2id = {}
    ilabel = 0
    for item in df_dataset["label"]:
        item = str(item)
        item = item.strip().lower()
        if item and item != "nan" and item not in category2id:
            category2id[item] = ilabel
            ilabel += 1
    return category2id

def save_category2id(category2id, category2id_path):
    with open(category2id_path, "w") as fw:
        s = sorted(category2id.items(), key=lambda e:e[1])
        for i in s:
            fw.write(i[0] + "\t" + str(i[1]) + "\n")


def load_category2id(category2id_path):
    category2id = {}
    with open(category2id_path, "r") as fr:
        for line in fr.readlines():
            line = line.strip().split("\t")
            category2id[line[0]] = int(line[1])
    return category2id


# generate train and test dataset, then map message to index
def split_train_test(df_dataset, word2id, split_rate, category2id):
    item2id = []
    max_len_sent = 0
    for i in df_dataset.index:
        label, message = df_dataset.loc[i].values[2], df_dataset.loc[i].values[3]
        if not isinstance(message, str):
            continue
        label = str(label)
        message = message.lower()
        label = category2id[label]
        message2id = [word2id.get(w, 0) for w in message.split()]
        item2id.append((label, message2id))
        if len(message2id) > max_len_sent:
            max_len_sent = len(message2id)
    random.shuffle(item2id)
    x_train, y_train, x_test, y_test = [], [], [], []
    for item in item2id[:int(len(item2id)*split_rate)]:
        x_test.append(item[1])
        y_test.append(item[0])
    for item in item2id[int(len(item2id)*split_rate):]:
        x_train.append(item[1])
        y_train.append(item[0])
    return x_train, y_train, x_test, y_test, max_len_sent

