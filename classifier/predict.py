# coding:utf-8
import argparse
import os
from cluster_to_classifier import TextClassifier
from keras.models import load_model
import data_preprocessing as dp
from keras.preprocessing import text, sequence
import numpy as np


def pad_or_truncate(sequence, max_length, padding_direction='pre', padding_item=0):
    if len(sequence) == max_length:
        return sequence
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        padding_count = (max_length - len(sequence))
        if padding_direction == 'pre':
            return [padding_item] * padding_count + sequence
        else:
            return sequence + [padding_item] * padding_count


nb_classes =159
model = load_model("../weights/weights.h5")
# discussions =["did not address my problem","Did not address issue with already created Apple ID","it doesnt address my requirement for diary issues"]
#discussions =["on the iPhone 7 differently opens with a click on icloud. I do not know what to do next.","iPhone not working","it does not work on iphone 5s phone"]
discussions =["I would like to know how to cancel my HBO subscription.",
              "I have a android now and need to cancel my subscription on apple and itunes.",
              "I need to cancel my subscription thru I tunes on my apps"]


discussions = [dp.denoise_text(text) for text in discussions]
word2id = dp.load_words("../dict/words.dict")
category2id = dp.load_category2id("../dict/category2id.dict")
print (category2id)

id2category = {item: i for i, item in category2id.items()}
print (id2category)

discussion_data = np.array(
    [pad_or_truncate([word2id.get(word, 0) for word in discussion.split()], 204)
     for discussion in discussions], dtype=np.int32)

y_preds = model.predict(discussion_data)
print("-------------------------\n")
for i in range(len(y_preds)):
    pred_label_id = np.argmax(y_preds[i])
    pred_label = str(id2category.get(pred_label_id,'None'))
    print('Message ' + str(i+1) + ':', discussions[i])
    print('Tag: ', pred_label)
    print('Score: ', y_preds[i][pred_label_id])
    print("-------------------------\n")