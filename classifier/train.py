# coding:utf-8
import argparse
import os
from cluster_to_classifier import TextClassifier

conf = "config.json"

# init model
model = TextClassifier(conf_path=conf, ispredict=0)

# training
model.train()
