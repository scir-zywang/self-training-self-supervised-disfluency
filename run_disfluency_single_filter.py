  # coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
import time
import pickle
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from itertools import chain
from transformers import ElectraConfig, BertTokenizer, AdamW
from transformers.modeling_electra import ElectraForSequenceDisfluency_sing
import torch.nn.functional as F
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

except_num = 0


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, disf_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the sentence pair. This should be
            specified for train and dev examples, but not for test examples.
            disf_label: (Optional) string. The label of the disfluency detection example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.disf_label = disf_label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_disf_id, label_id, label_sing_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_disf_id = label_disf_id
        self.label_sing_id = label_sing_id

class InputExample_eval(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, disf_label=None, sing_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the sentence pair. This should be
            specified for train and dev examples, but not for test examples.
            disf_label: (Optional) string. The label of the disfluency detection example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.disf_label = disf_label
        self.sing_label = sing_label

class InputFeatures_eval(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, label_disf_id, label_sing_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_disf_id = label_disf_id
        self.label_sing_id = label_sing_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

# &&&& 下面这个函数改了
class DisfluencyProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_pseudo_examples(self, data_dir):
        """See base class."""
        return self._create_pseudo_examples(
            self._read_tsv(os.path.join(data_dir, "pseudo.tsv")),
            "dev")
    # $$$ 修改了预训练任务1的labels
    def get_labels(self):
        """See base class."""
        return ["error_0", "error_1"]

    # $$$ 修改了序列标注labels
    def get_labels_disf(self):
        """See base class."""
        return ["O","D"]

    # $$$$ 添加了single prediction的label
    def get_single_disf(self):
        """See base class"""
        return ["wrong", "right"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        start_time = time.time()
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if i % 10000 == 0:
                logger.info("{} line data has been processed! used_time:{}".format(i, time.time() - start_time))
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            label_disf_id = line[4]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, disf_label=label_disf_id))
        return examples

    def _create_pseudo_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        start_time = time.time()
        examples = []
        for (i, line) in enumerate(lines):
            if i % 10000 == 0:
                logger.info("{} line data has been processed! used_time:{}".format(i, time.time() - start_time))
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            label_disf_id = line[4]
            text_a_fix = list()
            label_disf_id_fix = list()
            for i, j in zip(text_a.split(" "), label_disf_id.split(" ")):
                if j == "D":
                    continue
                text_a_fix.append(i)
                label_disf_id_fix.append(j)
            if len(label_disf_id_fix) == 0:
                continue
            examples.append(
                InputExample(guid=guid, text_a=" ".join(text_a_fix), text_b=text_b, label=label, disf_label=" ".join(label_disf_id_fix)))
        return examples


def label_to_map(label, label_map):
    label = label.strip().split(" ")
    out_label = []
    for el in label:
        out_label.append(label_map[el])
    return out_label

def random_word(text1, label, label_map, tokenizer, sel_prob):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :param label: labels such as ["D","O","O","D"]
    :param label_map: labels such as [0,1,,0]
    :param sel_prob: the prob to caluate the loss for each token
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    text = text1.strip().split(" ")
    orig_to_map_label = []
    orig_to_map_token = []
    if len(text) != len(label_map):
        print("text:{}".format(text))
        print("len text:{}".format(len(text)))
        print("label_map:{}".format(label_map))
        print("len label_map:{}".format(len(label_map)))
    assert len(text) == len(label_map)
    assert len(text) == len(label)

    for i in range(0, len(text)):
        orig_token = text[i]
        orig_label = label[i]
        orig_label_map = label_map[i]
        tokens = tokenizer.tokenize(orig_token)
        orig_to_map_token.extend(tokens)

        prob = random.random()
        if orig_label == "D":
            if prob < sel_prob:
                orig_to_map_label.append(orig_label_map)
            else:
                orig_to_map_label.append(-1)
        else:
            if prob < sel_prob / 5.0:
                orig_to_map_label.append(orig_label_map)
            else:
                orig_to_map_label.append(-1)

        for j in range(1, len(tokens)):
            orig_to_map_label.append(-1)
    # print(text1)
    # print (len(orig_to_map_label))
    # print (len(orig_to_map_token))
    # try:
    global except_num
    if len(orig_to_map_label) != len(orig_to_map_token):
        print(text1)
        except_num += 1
        if len(orig_to_map_token) > len(orig_to_map_label):
            orig_to_map_token = orig_to_map_token[:len(orig_to_map_label)]
        else:
            orig_to_map_label = orig_to_map_label[:len(orig_to_map_token)]
    assert len(orig_to_map_label) == len(orig_to_map_token)
    # except:
    #     except_num +=1
    return orig_to_map_token, orig_to_map_label


def random_word_lm(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_word_no_prob(text, label, label_map, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :param label: labels such as ["D","O","O","D"]
    :param label_map: labels such as [0,1,,0]
    :param sel_prob: the prob to caluate the loss for each token
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    text = text.replace("\n","").split(" ")
    orig_to_map_label = []
    orig_to_map_token = []
    assert len(text) == len(label_map)
    for i in range(0, len(text)):
        orig_token = text[i]
        # orig_label = label[i]
        orig_label_map = label_map[i]
        tokens = tokenizer.tokenize(orig_token)
        orig_to_map_token.extend(tokens)
        orig_to_map_label.append(orig_label_map)

        for j in range(1, len(tokens)):
            orig_to_map_label.append(-1)
    if len(orig_to_map_label) != len(orig_to_map_token):
        print(text)
        print(label_map)
        print(orig_to_map_label)
        print(orig_to_map_token)
        raise Exception(" ")
    # assert len(orig_to_map_label) == len(orig_to_map_token)
    return orig_to_map_token, orig_to_map_label


# &&&& 开始 下面这个函数改了
def convert_examples_to_features(examples, label_list, label_list_tagging, label_sing_list, max_seq_length, tokenizer, sel_prob,
                                 train_type="train"):
    start_time = time.time()
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    label_tagging_map = {label: i for i, label in enumerate(label_list_tagging)}
    label_sing_map = {label : i for i, label in enumerate(label_sing_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("{} lines of data have been converted, used time:{}".format(ex_index, time.time()-start_time))
        tokens_a = None
        tokens_b = None
        sing_label = None
        disf_label = None
        if example.text_b != "NONE":
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        elif example.text_b == "NONE" and example.disf_label == "NONE":
            tokens_a = tokenizer.tokenize(example.text_a)
            sing_label = example.label
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        else:
            # label_disf_id = label_tagging_map[example.disf_label]
            try:
                label_disf_id = label_to_map(example.disf_label, label_tagging_map)
            except:
                logger.info(example.disf_label)
                exit(0)
            if train_type == "train":
                tokens_a, disf_label = random_word(example.text_a, example.disf_label.strip().split(" "),
                                                   label_disf_id, tokenizer, 1000000)
            else:

                tokens_a, disf_label = random_word_no_prob(example.text_a, example.disf_label.strip().split(" "),
                                                           label_disf_id, tokenizer)
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                disf_label = disf_label[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        if tokens_b:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
            label_id = label_map[example.label]
            disf_label_id = [-1] * len(tokens)
            sing_label_id = -1
        elif example.text_b == "NONE" and example.disf_label == "NONE":
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            label_id = -1
            disf_label_id = [-1] * len(tokens)
            sing_label_id = label_sing_map[example.label]
        else:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            label_id = -1
            disf_label_id = ([-1] + disf_label + [-1])
            sing_label_id = -1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_disf = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        disf_label_id += padding_disf
        if len(input_ids) != max_seq_length:
            logger.info(len(input_ids))
            logger.info(max_seq_length)
            logger.info(input_ids)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(disf_label_id) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("disf_label_id: %s" % " ".join([str(x) for x in disf_label_id]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          label_disf_id=disf_label_id,
                          label_sing_id=sing_label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

# $$$ 替换为文本标错任务的评价函数
# def accuracy_tagging(eval_examples, predict_result_tagging, gold_result_tagging, input_mask_tagging, output_name):
#     output_file = open(output_name, "w")
#     example_id = -1
#     assert len(predict_result_tagging) == len(gold_result_tagging)
#     assert len(predict_result_tagging) == len(input_mask_tagging)
#     gold_number = 0
#     predict_number = 0
#     correct_number = 0
#     for i in range(0, len(predict_result_tagging)):
#         predict_results = predict_result_tagging[i]
#         gold_results = gold_result_tagging[i]
#         input_masks = input_mask_tagging[i]
#         assert len(predict_results) == len(gold_results)
#         assert len(predict_results) == len(input_masks)
#         for j in range(0, len(gold_results)):
#             example_id += 1
#             text_a = eval_examples[example_id].text_a.strip().split(" ")
#             length = input_masks[j].count(1)
#             # print (eval_examples[example_id].text_a)
#             # print (length)
#             gold_result_tmp = gold_results[j][0:length]
#             predict_result_tmp = predict_results[j][0:length]
#             gold_result_tmp = gold_result_tmp[1:len(gold_result_tmp) - 1]
#             predict_result_tmp = predict_result_tmp[1:len(predict_result_tmp) - 1]
#             assert len(gold_result_tmp) == len(predict_result_tmp)
#             gold_result = []
#             predict_result = []
#             for k in range(0, len(gold_result_tmp)):
#                 if gold_result_tmp[k] != -1:
#                     gold_result.append(gold_result_tmp[k])
#                     predict_result.append(predict_result_tmp[k])
#             assert len(text_a) == len(gold_result)
#
#             output_tokens = []
#             for l in range(0, len(text_a)):
#                 gold_label = "D" if gold_result[l] == 1 else "O"
#                 predict_label = "D" if predict_result[l] == 1 else "O"
#                 word = text_a[l]
#                 output_tokens.append(word + "#" + gold_label + "#" + predict_label)
#             output_file.write(" ".join(output_tokens) + "\n")
#
#             gold_number += gold_result.count(1)
#             predict_number += predict_result.count(1)
#             sum_result = list(map(lambda x: x[0] + x[1], zip(gold_result, predict_result)))
#             correct_number += sum_result.count(2)
#     # print (gold_result)
#     #         print (predict_result)
#     # print (gold_number)
#     # print (predict_number)
#     # print (correct_number)
#     output_file.close()
#     try:
#         p_score = correct_number * 1.0 / predict_number
#         r_score = correct_number * 1.0 / gold_number
#         f_score = 2.0 * p_score * r_score / (p_score + r_score)
#     except:
#         p_score = 0.0
#         r_score = 0.0
#         f_score = 0.0
#     return p_score, r_score, f_score
#
#
#
#
#
#
#
#
#     # for (ex_index, example) in enumerate(examples):
#     #         tokens_a = tokenizer.tokenize(example.text_a)
#     #         tokens_b = tokenizer.tokenize(example.text_b)
#
#
#
#     # return 0

# $$$ 增加文本检错任务的评价函数fpr
def fpr(pred_list, gold_list):
    fp = 0
    fp_tn = 0
    for pred, gold in zip(pred_list, gold_list):
        if gold[0] == "correct":
            fp_tn += 1
            if pred[0] != "correct":
                fp += 1
    if fp_tn == 0:
        return 0
    return round(fp / fp_tn, 4)


# $$$ 增加文本检错任务的评价函数detection_level
def detection_level(pred_list, gold_list):
    tp, fp, fn, tn = 0, 0, 0, 0
    for pred, gold in zip(pred_list, gold_list):
        if pred[0] == "correct":
            if gold[0] == "correct":
                tn += 1
            if gold[0] != "correct":
                fn += 1
        if pred[0] != "correct":
            if gold[0] == "correct":
                fp += 1
            if gold[0] != "correct":
                tp += 1
    acc = (tp + tn) / (tp + fp + tn + fn)
    if tp + fp != 0:
        p = tp / (tp + fp)
    else:
        p = 0
    if tp + fn != 0:
        r = tp / (tp + fn)
    else:
        r = 0
    if p + r != 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    return round(acc, 4), round(p, 4), round(r, 4), round(f1, 4)

# $$$ 增加文本检错任务的评价函数 iden_level
def iden_level(pred_list, gold_list):
    tn = 0
    tp = 0
    tp_fp = 0
    tp_fn = 0
    total = 0
    for pred, gold in zip(pred_list, gold_list):
        pred_id = set()
        gold_id = set()
        for id in pred:
            if id == "correct":
                pred_id.add(id)
            else:
                pred_id.add(id[0])
                if 'M' in id:
                    pred_id.add('M')
        for id in gold:
            if id == "correct":
                gold_id.add(id)
            else:
                gold_id.add(id[0])
                if 'M' in id:
                    gold_id.add('M')
        # print("pred_id:{}".format(pred_id))
        # print("gold_id:{}".format(gold_id))
        total += len(pred_id)
        if pred_id == gold_id:
            if list(pred_id)[0] == "correct":
                tn += 1
            else:
                tp += len(pred_id)
                tp_fp += len(pred_id)
                tp_fn += len(pred_id)
        if pred_id != gold_id:

            if list(pred_id)[0] != "correct":
                tp_fp += len(pred_id)
            if list(gold_id)[0] != "correct":
                tp_fn += len(gold_id)
            tp_set = pred_id & gold_id
            if "correct" not in tp_set:
                tp += len(tp_set)
    acc = (tp + tn) / total
    if tp_fp != 0:
        p = tp / tp_fp
    else:
        p = 0
    if tp_fn != 0:
        r = tp / tp_fn
    else:
        r = 0
    if p + r != 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    return round(acc, 4), round(p, 4), round(r, 4), round(f1, 4)


# $$$ 增加文本检错任务的评价函数 pos_level
def pos_level(pred_list, gold_list):
    pred_id = set()
    pred_id_correct = set()
    gold_id = set()
    gold_id_correct = set()
    for index, sub_list in enumerate(pred_list):
        for item in sub_list:
            if item == "correct":
                pred_id_correct.add("{}-{}".format(index, item))
            if item != "correct":
                pred_id.add("{}-{}".format(index, item))
    for index, sub_list in enumerate(gold_list):
        for item in sub_list:
            if item == "correct":
                gold_id_correct.add("{}-{}".format(index, item))
            if item != "correct":
                gold_id.add("{}-{}".format(index, item))
    tp = gold_id & pred_id
    tn = gold_id_correct & pred_id_correct

    if pred_id:
        p = len(tp) / len(pred_id)
    else:
        p = 0
    if gold_id:
        r = len(tp) / len(gold_id)
    else:
        r = 0
    if len(pred_id) + len(pred_id_correct) != 0:
        acc = (len(tp) + len(tn)) / (len(pred_id) + len(pred_id_correct))
    else:
        acc = 0
    if p + r != 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    return round(acc, 4), round(p, 4), round(r, 4), round(f1, 4)


# $$$ 添加计算得分的函数
def score_calculate(label_list, pred_output, input_mask, labels, output_loc):
    label_to_tag = {label: tag for label, tag in enumerate(label_list)}
    pred_labels = torch.argmax(pred_output, axis=2)
    pred_tags = [[label_to_tag[int(label)] for label in sent[1:int(input_len.sum()) - 1]] for input_len, sent in
                 zip(input_mask, pred_labels)]
    # for input_len, sent in zip(input_mask, labels):
    #     print("sent:{}".format(sent))
    #     print("sent_cut:{}".format(sent[1:input_len.sum()-1]))

    gold_tags = [[label_to_tag[int(label)] for label in sent[1:int(input_len.sum()) - 1]] for input_len, sent in
                 zip(input_mask, labels)]
    pred_span = convert_tags_to_span(pred_tags)

    gold_span = convert_tags_to_span(gold_tags)
    # for p, g in zip(pred_tags, gold_tags):
    #     print("pred:{} gold:{}".format(p, g))
    # for p, g in zip(pred_span, gold_span):
    #     print("pred:{} gold:{}".format(p, g))
    fpr_score = fpr(pred_list=pred_span, gold_list=gold_span)
    det_acc, det_p, det_r, det_f1 = detection_level(pred_list=pred_span, gold_list=gold_span)
    id_acc, id_p, id_r, id_f1 = iden_level(pred_list=pred_span, gold_list=gold_span)
    pos_acc, pos_p, pos_r, pos_f1 = pos_level(pred_list=pred_span, gold_list=gold_span)
    return fpr_score, det_acc, det_p, det_r, det_f1, id_acc, id_p, id_r, id_f1, pos_acc, pos_p, pos_r, pos_f1


# $$$ 添加将序列转换为span的函数，辅助评价函数计算。
def convert_tags_to_span(tags_data):
    spans = list()
    for sent in tags_data:
        sent_span = list()
        prev = 'C'
        prev_tag = 'C'
        begin = 0
        for index, tag in enumerate(sent):
            if tag == 'C':
                if prev == 'C':
                    continue
                if prev != 'C':
                    end = index
                    sent_span.append("{}-{}-{}".format(prev_tag, begin, end))
                    prev = 'C'
                    prev_tag = 'C'
                    continue
            if tag != 'C' and 'M' not in tag:
                if prev == 'C':
                    prev = tag
                    prev_tag = tag[-1]
                    begin = index + 1
                    continue
                if prev != 'C' and prev_tag == tag[-1] and tag[0] == "I":
                    prev = tag
                    continue
                if prev != 'C' and (prev_tag != tag[-1] or (prev_tag == tag[-1] and tag[0] == "B")):
                    end = index
                    sent_span.append("{}-{}-{}".format(prev_tag, begin, end))
                    prev = tag
                    prev_tag = tag[-1]
                    begin = index + 1
                    continue
            if tag != 'C' and tag == 'M':
                if prev == 'C':
                    begin = index + 1
                    end = index + 1
                    sent_span.append("{}-{}-{}".format(tag, begin, end))
                    continue
                if prev != 'C':
                    end = index
                    sent_span.append("{}-{}-{}".format(prev_tag, begin, end))
                    begin = index + 1
                    end = index + 1
                    sent_span.append("{}-{}-{}".format('M', begin, end))
                    prev_tag = "C"
                    prev = "C"
                    continue
            if tag != 'C' and tag != 'M' and 'M-' in tag:
                sent_span.append("{}-{}-{}".format('M', index + 1, index + 1))
                if prev == 'C':
                    prev = tag[2:]
                    prev_tag = tag[-1]
                    begin = index + 1
                    continue
                if prev != 'C' and prev_tag == tag[-1] and tag[-3] == "I":
                    prev = tag
                    continue
                if prev != 'C' and (prev_tag != tag[-1] or (prev_tag == tag[-1] and tag[2] == "B")):
                    end = index
                    sent_span.append("{}-{}-{}".format(prev_tag, begin, end))
                    prev = tag
                    prev_tag = tag[-1]
                    begin = index + 1
                    continue

        if prev != 'C':
            end = len(sent)
            sent_span.append("{}-{}-{}".format(prev_tag, begin, end))
        if not sent_span:
            sent_span.append("correct")
        spans.append(sent_span)
    return spans


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_tagging",
                        action='store_true',
                        help="Whether to run eval on the tagging set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--sel_prob",
                        default=0.5,
                        type=float,
                        help="The select prob for each word when pretraining.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--from_tf",
                        action='store_true',
                        help="from_tf")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--train_type',
                        type=str,
                        default="pretrain",
                        help="type of train")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # $$$ 此处修改
    parser.add_argument('--train_continue',
                        action='store_true',
                        help="continue train from the saved model")
    parser.add_argument('--saved_model_path',
                        type=str,
                        default=None,
                        help="The path of saved model trained before.")
    parser.add_argument('--use_new_model',
                        action='store_true',
                        help="read_feature_from_cache.")
    parser.add_argument('--feature_path',
                        type=str,
                        default=None,
                        help="The path of feature saved.")
    parser.add_argument("--pretrain_model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The  directory where the pretrain model comes")
    parser.add_argument("--pretrain_model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The  directory where the pretrain model comes")
    parser.add_argument("--thre",
                        default=0.5,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    # $$$
    args = parser.parse_args()

    # $$$ 改掉了task name
    processors = {
        "disfluency": DisfluencyProcessor,
    }

    # $$$ 改掉了task name
    num_labels_task = {
        "disfluency": 2,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()
    label_disf_list = processor.get_labels_disf()
    label_sing_list = processor.get_single_disf()
    num_labels_tagging = len(label_disf_list)
    pretrained = args.model_name_or_path

    tokenizer = BertTokenizer.from_pretrained(
        pretrained,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    state = None
    train_continued = False
    if args.train_continue:
        train_continued = True
    # if train_continued:
    #     state = torch.load(args.saved_model_path)
    #     model = BertForSequenceDisfluency.from_pretrained(args.bert_model,
    #                                                       state_dict=state['state_dict'],
    #                                                       num_labels=num_labels,
    #                                                       num_labels_tagging=num_labels_tagging)
    #     last_epoch = state['epoch']
    # else:
    #     config = ElectraConfig.from_pretrained(
    #         pretrained,
    #         num_labels=num_labels,
    #         finetuning_task=args.task_name,
    #         cache_dir=args.cache_dir if args.cache_dir else None,
    #     )
    #     model = ElectraForSequenceDisfluency_sing.from_pretrained(
    #         pretrained,
    #         config=config,
    #         cache_dir=args.cache_dir if args.cache_dir else None,
    #         num_labels=num_labels, num_labels_tagging=num_labels_tagging
    #     )
    if args.use_new_model:
        logger.info("对了")
        # new_model_file = os.path.join(args.pretrain_model_dir, "pytorch_model.bin")
        new_model_file = os.path.join(args.pretrain_model_dir, args.pretrain_model_name)
        logger.info("use pretrain model {}".format(new_model_file))
        state = torch.load(new_model_file)
        config = ElectraConfig.from_pretrained(
            pretrained,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = ElectraForSequenceDisfluency_sing.from_pretrained(
            pretrained,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
            state_dict=state['state_dict'],
            num_labels=num_labels, num_labels_tagging=num_labels_tagging
        )
    else:
        logger.info(
            "失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了失败了")
        config = ElectraConfig.from_pretrained(
            pretrained,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = ElectraForSequenceDisfluency_sing.from_pretrained(
            pretrained,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
            num_labels=num_labels, num_labels_tagging=num_labels_tagging
        )
    # if args.fp16:
    #     model.half()
    model.to(device)


    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        from apex import amp
        # optimizer = BertAdam(optimizer_grouped_parameters,
        #                      lr=args.learning_rate,
        #                      warmup=args.warmup_proportion,
        #                      t_total=t_total)
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        logger.info("amp handle init done!")
        # except ImportError:
        #     raise ImportError(
        #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if args.train_continue:
            optimizer.load_state_dict(state['optimizer'])

    # if args.local_rank != -1:
    #     try:
    #         from apex.parallel import DistributedDataParallel as DDP
    #     except ImportError:
    #         raise ImportError(
    #             "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #
    #     model = DDP(model)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_pseudo_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, label_disf_list, label_sing_list, args.max_seq_length, tokenizer, args.sel_prob,
            "dev")
        logger.info("***** Running evaluation on dev of epoch *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
        all_label_sing_ids = torch.tensor([f.label_sing_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                  all_label_disf_ids, all_label_sing_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

        model.eval()

        logits_total = list()
        input_ids_total = list()

        for input_ids, input_mask, segment_ids, label_ids, label_disf_ids, label_sing_ids in tqdm(eval_dataloader,
                                                                                                  desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            label_disf_ids = label_disf_ids.to(device)
            label_sing_ids = label_sing_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids=input_ids,
                                      token_type_ids=segment_ids,
                                      attention_mask=input_mask,
                                      labels_tagging=label_disf_ids)
                # &&&& 增加了sing的logits
                logits_pair, logits_tagging, logits_sing = model(input_ids=input_ids, token_type_ids=segment_ids
                                                                 , attention_mask=input_mask)

            # &&&&
            logits_sing = F.softmax(logits_sing, dim=-1)
            logits_sing = logits_sing.detach().cpu().numpy()
            logits_total.extend(logits_sing)
            input_ids_total.extend(input_ids.detach().cpu().numpy())

        # $$$
        output_eval_file = os.path.join(args.data_dir,
                                        "single_logits.tsv")
        with open(output_eval_file, "w") as f:
            for i in range(len(logits_total)):
                f.write(str(logits_total[i][0])+"\t"+str(logits_total[i][1])+"\n")

        with open(os.path.join(args.data_dir, "pseudo.tsv"), 'r', encoding='utf8') as f:
            t = f.readlines()

        with open(os.path.join(args.data_dir, "train.tsv"), 'w', encoding='utf8') as fw:
            for i in range(len(logits_total)):
                if logits_total[i][1] > args.thre:
                    fw.write(t[i].strip()+"\n")

        with open(os.path.join(args.data_dir, "single_input.tsv"), 'w', encoding='utf8') as fw:
            for e in eval_examples:
                fw.write(e.text_a+"\n")


        print("done!")
if __name__ == "__main__":
    main()
