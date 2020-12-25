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
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from itertools import chain
from transformers import ElectraConfig, BertTokenizer, AdamW
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from transformers.modeling_electra import ElectraForSequenceDisfluency_real

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def __init__(self, input_ids, input_mask, segment_ids, label_id, label_disf_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_disf_id = label_disf_id


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


class DisfluencyProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_unlabel_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "unlabel.tsv")),
            "unlabel")

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

    def get_fcic_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "fcic_test.tsv")),
            "fcic_test")

    def get_scotus_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "scotus_test.tsv")),
            "scotus_test")

    def get_callhome_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "callhome_test.tsv")),
            "callhome_test")

    def get_labels(self):
        """See base class."""
        return ["add_0", "add_1", "del_0", "del_1"]

    def get_labels_disf(self):
        """See base class."""
        return ["O", "D"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            disf_label = line[4]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, disf_label=disf_label))
        return examples


def label_to_map(label, label_map):
    label = label.strip().split(" ")
    out_label = []
    for el in label:
        out_label.append(label_map[el])
    return out_label


def random_word(text, label, label_map, tokenizer, sel_prob):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :param label: labels such as ["D","O","O","D"]
    :param label_map: labels such as [0,1,,0]
    :param sel_prob: the prob to caluate the loss for each token
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    text = text.strip().split(" ")
    orig_to_map_label = []
    orig_to_map_token = []

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
    assert len(orig_to_map_label) == len(orig_to_map_token)
    return orig_to_map_token, orig_to_map_label


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
    text = text.strip().split(" ")
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
    assert len(orig_to_map_label) == len(orig_to_map_token)
    return orig_to_map_token, orig_to_map_label

def random_word_no_prob_unlabel(text, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :param label: labels such as ["D","O","O","D"]
    :param label_map: labels such as [0,1,,0]
    :param sel_prob: the prob to caluate the loss for each token
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    text = text.strip().split(" ")
    orig_to_map_token = []
    orig_to_map_label = []
    for i in range(0, len(text)):
        orig_token = text[i]
        tokens = tokenizer.tokenize(orig_token)
        orig_to_map_token.extend(tokens)
        orig_to_map_label.append(1)
        for j in range(1, len(tokens)):
            orig_to_map_label.append(-1)
    assert len(orig_to_map_label) == len(orig_to_map_token)
    return orig_to_map_token, orig_to_map_label


def convert_examples_to_features(examples, label_list, label_list_tagging, max_seq_length, tokenizer, sel_prob,
                                 train_type="train"):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    label_tagging_map = {label: i for i, label in enumerate(label_list_tagging)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = None
        tokens_b = None
        if example.text_b != "NONE":
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # label_disf_id = label_tagging_map[example.disf_label]
            label_disf_id = label_to_map(example.disf_label, label_tagging_map)
            # if train_type == "train":
            #     tokens_a, disf_label = random_word(example.text_a, example.disf_label.strip().split(" "),
            #                                        label_disf_id, tokenizer, sel_prob)
            # else:
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
        else:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            label_id = -1
            disf_label_id = ([-1] + disf_label + [-1])

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

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(disf_label_id) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("disf_label_id: %s" % " ".join([str(x) for x in disf_label_id]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          label_disf_id=disf_label_id))
    return features

def convert_examples_to_features_unlabel(examples, max_seq_length, tokenizer, sel_prob,
                                 train_type="train"):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens_a, disf_label = random_word_no_prob_unlabel(example.text_a, tokenizer)
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

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        disf_label_id = ([-1] + disf_label + [-1])
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

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(disf_label_id) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("disf_label_id: %s" % " ".join([str(x) for x in disf_label_id]))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_disf_id=disf_label_id,
                          label_id=None))
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


def unlabel_tagging(eval_examples, predict_result_tagging, gold_result_tagging, input_mask_tagging, output_name):
    output_file = open(output_name, "w")
    example_id = -1
    assert len(predict_result_tagging) == len(input_mask_tagging)
    id = 0
    for i in range(0, len(predict_result_tagging)):
        predict_results = predict_result_tagging[i]
        gold_results = gold_result_tagging[i]
        input_masks = input_mask_tagging[i]
        assert len(predict_results) == len(input_masks)
        for j in range(0, len(predict_results)):
            example_id += 1
            text_a = eval_examples[example_id].text_a.strip().split(" ")
            length = input_masks[j].count(1)

            gold_result_tmp = gold_results[j][0:length]
            predict_result_tmp = predict_results[j][0:length]
            gold_result_tmp = gold_result_tmp[1:len(gold_result_tmp) - 1]
            predict_result_tmp = predict_result_tmp[1:len(predict_result_tmp) - 1]
            gold_result = []
            predict_result = []
            for k in range(0, len(gold_result_tmp)):
                if gold_result_tmp[k] != -1:
                    gold_result.append(gold_result_tmp[k])
                    predict_result.append(predict_result_tmp[k])

            assert len(text_a) == len(predict_result)

            output_tokens = []
            for l in range(0, len(text_a)):
                predict_label = "D" if predict_result[l] == 1 else "O"
                output_tokens.append(predict_label)
            output_file.write(str(id)+"\t"+" ".join(text_a)+"\tNONE\tNONE\t"+" ".join(output_tokens) + "\n")
            id += 1
    output_file.close()



def accuracy_tagging(eval_examples, predict_result_tagging, gold_result_tagging, input_mask_tagging, output_name):
    output_file = open(output_name, "w")
    example_id = -1
    assert len(predict_result_tagging) == len(gold_result_tagging)
    assert len(predict_result_tagging) == len(input_mask_tagging)
    gold_number = 0
    predict_number = 0
    correct_number = 0
    for i in range(0, len(predict_result_tagging)):
        predict_results = predict_result_tagging[i]
        gold_results = gold_result_tagging[i]
        input_masks = input_mask_tagging[i]
        assert len(predict_results) == len(gold_results)
        assert len(predict_results) == len(input_masks)
        for j in range(0, len(gold_results)):
            example_id += 1
            text_a = eval_examples[example_id].text_a.strip().split(" ")
            length = input_masks[j].count(1)
            # print (eval_examples[example_id].text_a)
            # print (length)
            gold_result_tmp = gold_results[j][0:length]
            predict_result_tmp = predict_results[j][0:length]
            gold_result_tmp = gold_result_tmp[1:len(gold_result_tmp) - 1]
            predict_result_tmp = predict_result_tmp[1:len(predict_result_tmp) - 1]
            assert len(gold_result_tmp) == len(predict_result_tmp)
            gold_result = []
            predict_result = []
            for k in range(0, len(gold_result_tmp)):
                if gold_result_tmp[k] != -1:
                    gold_result.append(gold_result_tmp[k])
                    predict_result.append(predict_result_tmp[k])
            assert len(text_a) == len(gold_result)

            output_tokens = []
            for l in range(0, len(text_a)):
                gold_label = "D" if gold_result[l] == 1 else "O"
                predict_label = "D" if predict_result[l] == 1 else "O"
                word = text_a[l]
                output_tokens.append(word + "#" + gold_label + "#" + predict_label)
            output_file.write(" ".join(output_tokens) + "\n")

            gold_number += gold_result.count(1)
            predict_number += predict_result.count(1)
            sum_result = list(map(lambda x: x[0] + x[1], zip(gold_result, predict_result)))
            correct_number += sum_result.count(2)
    # print (gold_result)
    #         print (predict_result)
    # print (gold_number)
    # print (predict_number)
    # print (correct_number)
    output_file.close()
    try:
        p_score = correct_number * 1.0 / predict_number
        r_score = correct_number * 1.0 / gold_number
        f_score = 2.0 * p_score * r_score / (p_score + r_score)
    except:
        p_score = 0
        r_score = 0
        f_score = 0
    return p_score, r_score, f_score








    # for (ex_index, example) in enumerate(examples):
    #         tokens_a = tokenizer.tokenize(example.text_a)
    #         tokens_b = tokenizer.tokenize(example.text_b)



    # return 0


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
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
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
    parser.add_argument("--do_unlabel",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_format",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_tagging",
                        action='store_true',
                        help="Whether to run eval on the tagging set.")
    parser.add_argument("--use_new_model",
                        action='store_true',
                        help="use new model instead of google model to ini when training")
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
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--unlabel_size',
                        type=int,
                        default=500,
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

    args = parser.parse_args()

    processors = {
        "disfluency": DisfluencyProcessor,
    }

    num_labels_task = {
        "disfluency": 4,
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

    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()
    label_disf_list = processor.get_labels_disf()
    num_labels_tagging = len(label_disf_list)

    pretrained = str(args.model_name_or_path)

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
        if "state_dict" in state:
            state = state['state_dict']
        model = ElectraForSequenceDisfluency_real.from_pretrained(
            pretrained,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
            state_dict=state,
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
        model = ElectraForSequenceDisfluency_real.from_pretrained(
            pretrained,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
            num_labels=num_labels, num_labels_tagging=num_labels_tagging
        )
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

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
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        prev_best_dev_f1 = -1.0
        prev_best_test_f1 = -1.0
        train_features = convert_examples_to_features(
            train_examples, label_list, label_disf_list, args.max_seq_length, tokenizer, args.sel_prob, "train")
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_label_disf_ids = torch.tensor([f.label_disf_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_disf_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # model.train()
        epoch_size = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            epoch_size += 1
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, label_disf_ids = batch
                loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels_tagging=label_disf_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1



            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = processor.get_dev_examples(args.data_dir)
                eval_features = convert_examples_to_features(
                    eval_examples, label_list, label_disf_list, args.max_seq_length, tokenizer, args.sel_prob, "dev")
                logger.info("***** Running evaluation on dev of epoch %d *****", epoch_size)
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.train_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                          all_label_disf_ids)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                # predict_result_pair = []
                predict_result_tagging = []
                # gold_result_pair = []
                gold_result_tagging = []
                input_mask_tagging = []

                for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader,
                                                                                          desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    label_disf_ids = label_disf_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids=input_ids,
                                              token_type_ids=segment_ids,
                                              attention_mask=input_mask,
                                              labels_tagging=label_disf_ids)
                        logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                            attention_mask=input_mask)

                    logits_pair = logits_pair.detach().cpu().numpy()
                    logits_tagging = logits_tagging.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    label_disf_ids = label_disf_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()

                    if args.do_tagging:
                        predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
                        gold_result_tagging.append(label_disf_ids.tolist())
                        input_mask_tagging.append(input_mask.tolist())
                    else:
                        tmp_eval_accuracy = accuracy(logits_pair, label_ids)
                        # print (np.argmax(logits_pair, axis=1))
                        # print (logits_pair)
                        # print (label_ids)
                        eval_accuracy += tmp_eval_accuracy

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1
                eval_loss = eval_loss / nb_eval_steps
                if args.do_tagging:
                    p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging,
                                                                 gold_result_tagging, input_mask_tagging,
                                                                 os.path.join(args.output_dir,
                                                                              "dev_results.txt.epoch" + str(
                                                                                  epoch_size)))
                    result = {'eval_loss': eval_loss,
                              'dev p_score': p_score,
                              'dev r_score': r_score,
                              'dev f_score': f_score}
                    if f_score > prev_best_dev_f1:
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                        # if args.do_train:
                        output_eval_file = os.path.join(args.output_dir, "best_dev.epoch" + str(epoch_size))
                        with open(output_eval_file, "w") as writer:
                            writer.write("best")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        prev_best_dev_f1 = f_score


                    output_eval_file = os.path.join(args.output_dir, "dev_eval_results.txt.epoch" + str(epoch_size))
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Dev Eval results %d*****", epoch_size)
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))


                            # for key in sorted(result.keys()):
                            #     logger.info("  %s = %s", key, str(result[key]))
                else:
                    eval_accuracy = eval_accuracy / nb_eval_examples
                    loss = tr_loss / nb_tr_steps if args.do_train else None
                    result = {'dev eval_loss': eval_loss,
                              'dev eval_accuracy': eval_accuracy,
                              'dev global_step': global_step,
                              'dev loss': loss}
                    output_eval_file = os.path.join(args.output_dir, "dev_eval_results.txt.epoch" + str(epoch_size))
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Eval results %d*****", epoch_size)
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))

            if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = processor.get_test_examples(args.data_dir)
                eval_features = convert_examples_to_features(
                    eval_examples, label_list, label_disf_list, args.max_seq_length, tokenizer, args.sel_prob, "test")
                logger.info("***** Running evaluation on test %d*****", epoch_size)
                logger.info("  Test Num examples = %d", len(eval_examples))
                logger.info("  Test Batch size = %d", args.train_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                          all_label_disf_ids)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                # predict_result_pair = []
                predict_result_tagging = []
                # gold_result_pair = []
                gold_result_tagging = []
                input_mask_tagging = []

                for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader,
                                                                                          desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    label_disf_ids = label_disf_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids=input_ids,
                                              token_type_ids=segment_ids,
                                              attention_mask=input_mask,
                                              labels_tagging=label_disf_ids)
                        logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                            attention_mask=input_mask)

                    logits_pair = logits_pair.detach().cpu().numpy()
                    logits_tagging = logits_tagging.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    label_disf_ids = label_disf_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()

                    if args.do_tagging:
                        predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
                        gold_result_tagging.append(label_disf_ids.tolist())
                        input_mask_tagging.append(input_mask.tolist())
                    else:
                        tmp_eval_accuracy = accuracy(logits_pair, label_ids)
                        # print (np.argmax(logits_pair, axis=1))
                        # print (logits_pair)
                        # print (label_ids)
                        eval_accuracy += tmp_eval_accuracy

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1
                eval_loss = eval_loss / nb_eval_steps
                if args.do_tagging:
                    p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging,
                                                                 gold_result_tagging, input_mask_tagging,
                                                                 os.path.join(args.output_dir,
                                                                              "test_results.txt.epoch" + str(
                                                                                  epoch_size)))
                    result = {'test_loss': eval_loss,
                              'test p_score': p_score,
                              'test r_score': r_score,
                              'test f_score': f_score}
                    output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt.epoch" + str(epoch_size))
                    if f_score > prev_best_test_f1:
                        prev_best_test_f1 = f_score
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Test Eval results epoch%d*****", epoch_size)
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                else:
                    eval_accuracy = eval_accuracy / nb_eval_examples
                    loss = tr_loss / nb_tr_steps if args.do_train else None
                    result = {'test eval_loss': eval_loss,
                              'test eval_accuracy': eval_accuracy,
                              'test global_step': global_step,
                              'test loss': loss}
                    output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt.epoch" + str(epoch_size))
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Eval results *****")
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
            # if epoch_size == 1:
            #     if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            #         eval_examples = processor.get_scotus_test_examples(args.data_dir)
            #         eval_features = convert_examples_to_features(
            #             eval_examples, label_list, label_disf_list, args.max_seq_length, tokenizer, args.sel_prob, "scotus")
            #         logger.info("***** Running evaluation on scotus %d*****", epoch_size)
            #         logger.info("  Test Num examples = %d", len(eval_examples))
            #         logger.info("  Test Batch size = %d", args.train_batch_size)
            #         all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            #         all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            #         all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            #         all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            #         all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
            #         eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
            #                                   all_label_disf_ids)
            #         # Run prediction for full data
            #         eval_sampler = SequentialSampler(eval_data)
            #         eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)
            #
            #         model.eval()
            #         eval_loss, eval_accuracy = 0, 0
            #         nb_eval_steps, nb_eval_examples = 0, 0
            #
            #         # predict_result_pair = []
            #         predict_result_tagging = []
            #         # gold_result_pair = []
            #         gold_result_tagging = []
            #         input_mask_tagging = []
            #
            #         for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader,
            #                                                                                   desc="Evaluating"):
            #             input_ids = input_ids.to(device)
            #             input_mask = input_mask.to(device)
            #             segment_ids = segment_ids.to(device)
            #             label_ids = label_ids.to(device)
            #             label_disf_ids = label_disf_ids.to(device)
            #
            #             with torch.no_grad():
            #                 tmp_eval_loss = model(input_ids=input_ids,
            #                                       token_type_ids=segment_ids,
            #                                       attention_mask=input_mask,
            #                                       labels_tagging=label_disf_ids)
            #                 logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
            #                                                     attention_mask=input_mask)
            #
            #             logits_pair = logits_pair.detach().cpu().numpy()
            #             logits_tagging = logits_tagging.detach().cpu().numpy()
            #             label_ids = label_ids.to('cpu').numpy()
            #             label_disf_ids = label_disf_ids.to('cpu').numpy()
            #             input_mask = input_mask.to('cpu').numpy()
            #
            #             if args.do_tagging:
            #                 predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
            #                 gold_result_tagging.append(label_disf_ids.tolist())
            #                 input_mask_tagging.append(input_mask.tolist())
            #             else:
            #                 tmp_eval_accuracy = accuracy(logits_pair, label_ids)
            #                 # print (np.argmax(logits_pair, axis=1))
            #                 # print (logits_pair)
            #                 # print (label_ids)
            #                 eval_accuracy += tmp_eval_accuracy
            #
            #             eval_loss += tmp_eval_loss.mean().item()
            #             nb_eval_examples += input_ids.size(0)
            #             nb_eval_steps += 1
            #         eval_loss = eval_loss / nb_eval_steps
            #         if args.do_tagging:
            #             p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging,
            #                                                          gold_result_tagging, input_mask_tagging,
            #                                                          os.path.join(args.output_dir,
            #                                                                       "scotus_results.txt.epoch" + str(
            #                                                                           epoch_size)))
            #             result = {'test_loss': eval_loss,
            #                       'test p_score': p_score,
            #                       'test r_score': r_score,
            #                       'test f_score': f_score}
            #             output_eval_file = os.path.join(args.output_dir, "scotus_eval_results.txt.epoch" + str(epoch_size))
            #             if f_score > prev_best_test_f1:
            #                 prev_best_test_f1 = f_score
            #             with open(output_eval_file, "w") as writer:
            #                 logger.info("***** Scotus Eval results epoch%d*****", epoch_size)
            #                 for key in sorted(result.keys()):
            #                     logger.info("  %s = %s", key, str(result[key]))
            #                     writer.write("%s = %s\n" % (key, str(result[key])))
            #         else:
            #             eval_accuracy = eval_accuracy / nb_eval_examples
            #             loss = tr_loss / nb_tr_steps if args.do_train else None
            #             result = {'test eval_loss': eval_loss,
            #                       'test eval_accuracy': eval_accuracy,
            #                       'test global_step': global_step,
            #                       'test loss': loss}
            #             output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt.epoch" + str(epoch_size))
            #             with open(output_eval_file, "w") as writer:
            #                 logger.info("***** Eval results *****")
            #                 for key in sorted(result.keys()):
            #                     logger.info("  %s = %s", key, str(result[key]))
            #                     writer.write("%s = %s\n" % (key, str(result[key])))
            #
            #
            #     if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            #         eval_examples = processor.get_fcic_test_examples(args.data_dir)
            #         eval_features = convert_examples_to_features(
            #             eval_examples, label_list, label_disf_list, args.max_seq_length, tokenizer, args.sel_prob, "fcic")
            #         logger.info("***** Running evaluation on fcic %d*****", epoch_size)
            #         logger.info("  Test Num examples = %d", len(eval_examples))
            #         logger.info("  Test Batch size = %d", args.train_batch_size)
            #         all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            #         all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            #         all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            #         all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            #         all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
            #         eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
            #                                   all_label_disf_ids)
            #         # Run prediction for full data
            #         eval_sampler = SequentialSampler(eval_data)
            #         eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)
            #
            #         model.eval()
            #         eval_loss, eval_accuracy = 0, 0
            #         nb_eval_steps, nb_eval_examples = 0, 0
            #
            #         # predict_result_pair = []
            #         predict_result_tagging = []
            #         # gold_result_pair = []
            #         gold_result_tagging = []
            #         input_mask_tagging = []
            #
            #         for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader,
            #                                                                                   desc="Evaluating"):
            #             input_ids = input_ids.to(device)
            #             input_mask = input_mask.to(device)
            #             segment_ids = segment_ids.to(device)
            #             label_ids = label_ids.to(device)
            #             label_disf_ids = label_disf_ids.to(device)
            #
            #             with torch.no_grad():
            #                 tmp_eval_loss = model(input_ids=input_ids,
            #                                       token_type_ids=segment_ids,
            #                                       attention_mask=input_mask,
            #                                       labels_tagging=label_disf_ids)
            #                 logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
            #                                                     attention_mask=input_mask)
            #
            #             logits_pair = logits_pair.detach().cpu().numpy()
            #             logits_tagging = logits_tagging.detach().cpu().numpy()
            #             label_ids = label_ids.to('cpu').numpy()
            #             label_disf_ids = label_disf_ids.to('cpu').numpy()
            #             input_mask = input_mask.to('cpu').numpy()
            #
            #             if args.do_tagging:
            #                 predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
            #                 gold_result_tagging.append(label_disf_ids.tolist())
            #                 input_mask_tagging.append(input_mask.tolist())
            #             else:
            #                 tmp_eval_accuracy = accuracy(logits_pair, label_ids)
            #                 # print (np.argmax(logits_pair, axis=1))
            #                 # print (logits_pair)
            #                 # print (label_ids)
            #                 eval_accuracy += tmp_eval_accuracy
            #
            #             eval_loss += tmp_eval_loss.mean().item()
            #             nb_eval_examples += input_ids.size(0)
            #             nb_eval_steps += 1
            #         eval_loss = eval_loss / nb_eval_steps
            #         if args.do_tagging:
            #             p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging,
            #                                                          gold_result_tagging, input_mask_tagging,
            #                                                          os.path.join(args.output_dir,
            #                                                                       "fcic_results.txt.epoch" + str(
            #                                                                           epoch_size)))
            #             result = {'test_loss': eval_loss,
            #                       'test p_score': p_score,
            #                       'test r_score': r_score,
            #                       'test f_score': f_score}
            #             output_eval_file = os.path.join(args.output_dir, "fcic_eval_results.txt.epoch" + str(epoch_size))
            #             if f_score > prev_best_test_f1:
            #                 prev_best_test_f1 = f_score
            #             with open(output_eval_file, "w") as writer:
            #                 logger.info("***** Fcic Eval results epoch%d*****", epoch_size)
            #                 for key in sorted(result.keys()):
            #                     logger.info("  %s = %s", key, str(result[key]))
            #                     writer.write("%s = %s\n" % (key, str(result[key])))
            #         else:
            #             eval_accuracy = eval_accuracy / nb_eval_examples
            #             loss = tr_loss / nb_tr_steps if args.do_train else None
            #             result = {'test eval_loss': eval_loss,
            #                       'test eval_accuracy': eval_accuracy,
            #                       'test global_step': global_step,
            #                       'test loss': loss}
            #             output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt.epoch" + str(epoch_size))
            #             with open(output_eval_file, "w") as writer:
            #                 logger.info("***** Eval results *****")
            #                 for key in sorted(result.keys()):
            #                     logger.info("  %s = %s", key, str(result[key]))
            #                     writer.write("%s = %s\n" % (key, str(result[key])))
            #
            #     if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            #         eval_examples = processor.get_callhome_test_examples(args.data_dir)
            #         eval_features = convert_examples_to_features(
            #             eval_examples, label_list, label_disf_list, args.max_seq_length, tokenizer, args.sel_prob, "callhome")
            #         logger.info("***** Running evaluation on callhome %d*****", epoch_size)
            #         logger.info("  Test Num examples = %d", len(eval_examples))
            #         logger.info("  Test Batch size = %d", args.train_batch_size)
            #         all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            #         all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            #         all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            #         all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            #         all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
            #         eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
            #                                   all_label_disf_ids)
            #         # Run prediction for full data
            #         eval_sampler = SequentialSampler(eval_data)
            #         eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)
            #
            #         model.eval()
            #         eval_loss, eval_accuracy = 0, 0
            #         nb_eval_steps, nb_eval_examples = 0, 0
            #
            #         # predict_result_pair = []
            #         predict_result_tagging = []
            #         # gold_result_pair = []
            #         gold_result_tagging = []
            #         input_mask_tagging = []
            #
            #         for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader,
            #                                                                                   desc="Evaluating"):
            #             input_ids = input_ids.to(device)
            #             input_mask = input_mask.to(device)
            #             segment_ids = segment_ids.to(device)
            #             label_ids = label_ids.to(device)
            #             label_disf_ids = label_disf_ids.to(device)
            #
            #             with torch.no_grad():
            #                 tmp_eval_loss = model(input_ids=input_ids,
            #                                       token_type_ids=segment_ids,
            #                                       attention_mask=input_mask,
            #                                       labels_tagging=label_disf_ids)
            #                 logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
            #                                                     attention_mask=input_mask)
            #
            #             logits_pair = logits_pair.detach().cpu().numpy()
            #             logits_tagging = logits_tagging.detach().cpu().numpy()
            #             label_ids = label_ids.to('cpu').numpy()
            #             label_disf_ids = label_disf_ids.to('cpu').numpy()
            #             input_mask = input_mask.to('cpu').numpy()
            #
            #             if args.do_tagging:
            #                 predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
            #                 gold_result_tagging.append(label_disf_ids.tolist())
            #                 input_mask_tagging.append(input_mask.tolist())
            #             else:
            #                 tmp_eval_accuracy = accuracy(logits_pair, label_ids)
            #                 # print (np.argmax(logits_pair, axis=1))
            #                 # print (logits_pair)
            #                 # print (label_ids)
            #                 eval_accuracy += tmp_eval_accuracy
            #
            #             eval_loss += tmp_eval_loss.mean().item()
            #             nb_eval_examples += input_ids.size(0)
            #             nb_eval_steps += 1
            #         eval_loss = eval_loss / nb_eval_steps
            #         if args.do_tagging:
            #             p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging,
            #                                                          gold_result_tagging, input_mask_tagging,
            #                                                          os.path.join(args.output_dir,
            #                                                                       "callhome_results.txt.epoch" + str(
            #                                                                           epoch_size)))
            #             result = {'test_loss': eval_loss,
            #                       'test p_score': p_score,
            #                       'test r_score': r_score,
            #                       'test f_score': f_score}
            #             output_eval_file = os.path.join(args.output_dir, "callhome_eval_results.txt.epoch" + str(epoch_size))
            #             if f_score > prev_best_test_f1:
            #                 prev_best_test_f1 = f_score
            #             with open(output_eval_file, "w") as writer:
            #                 logger.info("***** Callhome Eval results epoch%d*****", epoch_size)
            #                 for key in sorted(result.keys()):
            #                     logger.info("  %s = %s", key, str(result[key]))
            #                     writer.write("%s = %s\n" % (key, str(result[key])))
            #         else:
            #             eval_accuracy = eval_accuracy / nb_eval_examples
            #             loss = tr_loss / nb_tr_steps if args.do_train else None
            #             result = {'test eval_loss': eval_loss,
            #                       'test eval_accuracy': eval_accuracy,
            #                       'test global_step': global_step,
            #                       'test loss': loss}
            #             output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt.epoch" + str(epoch_size))
            #             with open(output_eval_file, "w") as writer:
            #                 logger.info("***** Eval results *****")
            #                 for key in sorted(result.keys()):
            #                     logger.info("  %s = %s", key, str(result[key]))
            #                     writer.write("%s = %s\n" % (key, str(result[key])))

    # &&&& 给数据打标签
    if args.do_unlabel and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        unlabel_examples = processor.get_unlabel_examples(args.data_dir)
        random.shuffle(unlabel_examples)
        unlabel_examples = unlabel_examples[:args.unlabel_size]
        with open(os.path.join(args.output_dir, "unlabel_gold.txt"), 'w') as f:
            for index, line in enumerate(unlabel_examples):
                f.write(str(
                    index) + "\t" + line.text_a + "\t" + line.text_b + "\t" + line.label + "\t" + line.disf_label + "\n")
        unlabel_features = convert_examples_to_features_unlabel(
            unlabel_examples, args.max_seq_length, tokenizer, args.sel_prob, "unlabel")
        logger.info("***** Running evaluation on dev of epoch *****")
        logger.info("  Num examples = %d", len(unlabel_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in unlabel_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in unlabel_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in unlabel_features], dtype=torch.long)
        all_label_disf_ids = torch.tensor([f.label_disf_id for f in unlabel_features],
                                          dtype=torch.long)
        unlabel_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                     all_label_disf_ids)
        unlabel_sampler = SequentialSampler(unlabel_data)
        unlabel_dataloader = DataLoader(unlabel_data, sampler=unlabel_sampler,
                                        batch_size=args.train_batch_size)

        model.eval()

        predict_result_tagging = []
        gold_result_tagging = []
        input_mask_tagging = []

        for input_ids, input_mask, segment_ids, label_disf_ids in tqdm(unlabel_dataloader,
                                                                       desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_disf_ids = label_disf_ids.to(device)

            with torch.no_grad():
                logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                    attention_mask=input_mask)

            logits_tagging = logits_tagging.detach().cpu().numpy()
            input_mask = input_mask.to('cpu').numpy()
            label_disf_ids = label_disf_ids.to('cpu').numpy()

            gold_result_tagging.append(label_disf_ids.tolist())
            predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
            input_mask_tagging.append(input_mask.tolist())

        unlabel_tagging(unlabel_examples, predict_result_tagging, gold_result_tagging,
                        input_mask_tagging,
                        os.path.join(args.output_dir,
                                     "unlabel_results.txt"))

    # &&&& 给数据打标签
    if args.do_eval_format and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        unlabel_examples = processor.get_dev_examples(args.data_dir)
        unlabel_features = convert_examples_to_features_unlabel(
            unlabel_examples, args.max_seq_length, tokenizer, args.sel_prob, "unlabel")
        logger.info("***** Running evaluation on dev of epoch *****")
        logger.info("  Num examples = %d", len(unlabel_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in unlabel_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in unlabel_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in unlabel_features], dtype=torch.long)
        all_label_disf_ids = torch.tensor([f.label_disf_id for f in unlabel_features],
                                          dtype=torch.long)
        unlabel_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                     all_label_disf_ids)
        unlabel_sampler = SequentialSampler(unlabel_data)
        unlabel_dataloader = DataLoader(unlabel_data, sampler=unlabel_sampler,
                                        batch_size=args.train_batch_size)

        model.eval()

        predict_result_tagging = []
        gold_result_tagging = []
        input_mask_tagging = []

        for input_ids, input_mask, segment_ids, label_disf_ids in tqdm(unlabel_dataloader,
                                                                       desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_disf_ids = label_disf_ids.to(device)

            with torch.no_grad():
                logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                    attention_mask=input_mask)

            logits_tagging = logits_tagging.detach().cpu().numpy()
            input_mask = input_mask.to('cpu').numpy()
            label_disf_ids = label_disf_ids.to('cpu').numpy()

            gold_result_tagging.append(label_disf_ids.tolist())
            predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
            input_mask_tagging.append(input_mask.tolist())

        unlabel_tagging(unlabel_examples, predict_result_tagging, gold_result_tagging,
                        input_mask_tagging,
                        os.path.join(args.output_dir,
                                     "dev_format.txt"))

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, label_disf_list, args.max_seq_length, tokenizer, args.sel_prob, "dev")
        logger.info("***** Running evaluation on dev of epoch *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                  all_label_disf_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # predict_result_pair = []
        predict_result_tagging = []
        # gold_result_pair = []
        gold_result_tagging = []
        input_mask_tagging = []

        for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader,
                                                                                  desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            label_disf_ids = label_disf_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids=input_ids,
                                      token_type_ids=segment_ids,
                                      attention_mask=input_mask,
                                      labels_tagging=label_disf_ids)
                logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                    attention_mask=input_mask)

            logits_pair = logits_pair.detach().cpu().numpy()
            logits_tagging = logits_tagging.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            label_disf_ids = label_disf_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            if args.do_tagging:
                predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
                gold_result_tagging.append(label_disf_ids.tolist())
                input_mask_tagging.append(input_mask.tolist())
            else:
                tmp_eval_accuracy = accuracy(logits_pair, label_ids)
                # print (np.argmax(logits_pair, axis=1))
                # print (logits_pair)
                # print (label_ids)
                eval_accuracy += tmp_eval_accuracy

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        if args.do_tagging:
            p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging,
                                                         gold_result_tagging, input_mask_tagging,
                                                         os.path.join(args.output_dir,
                                                                      "dev_results.txt.final"))
            result = {'eval_loss': eval_loss,
                      'dev p_score': p_score,
                      'dev r_score': r_score,
                      'dev f_score': f_score}

            output_eval_file = os.path.join(args.output_dir, "dev_eval_results.txt.final")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Dev Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

                    # for key in sorted(result.keys()):
                    #     logger.info("  %s = %s", key, str(result[key]))
            if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = processor.get_test_examples(args.data_dir)
                eval_features = convert_examples_to_features(
                    eval_examples, label_list, label_disf_list, args.max_seq_length, tokenizer, args.sel_prob, "test")
                logger.info("***** Running evaluation on test *****",)
                logger.info("  Test Num examples = %d", len(eval_examples))
                logger.info("  Test Batch size = %d", args.train_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                          all_label_disf_ids)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                # predict_result_pair = []
                predict_result_tagging = []
                # gold_result_pair = []
                gold_result_tagging = []
                input_mask_tagging = []

                for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader,
                                                                                          desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    label_disf_ids = label_disf_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids=input_ids,
                                              token_type_ids=segment_ids,
                                              attention_mask=input_mask,
                                              labels_tagging=label_disf_ids)
                        logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                            attention_mask=input_mask)

                    logits_pair = logits_pair.detach().cpu().numpy()
                    logits_tagging = logits_tagging.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    label_disf_ids = label_disf_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()

                    if args.do_tagging:
                        predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
                        gold_result_tagging.append(label_disf_ids.tolist())
                        input_mask_tagging.append(input_mask.tolist())
                    else:
                        tmp_eval_accuracy = accuracy(logits_pair, label_ids)
                        # print (np.argmax(logits_pair, axis=1))
                        # print (logits_pair)
                        # print (label_ids)
                        eval_accuracy += tmp_eval_accuracy

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1
                eval_loss = eval_loss / nb_eval_steps
                if args.do_tagging:
                    p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging,
                                                                 gold_result_tagging, input_mask_tagging,
                                                                 os.path.join(args.output_dir,
                                                                              "test_results.txt.final"))
                    result = {'test_loss': eval_loss,
                              'test p_score': p_score,
                              'test r_score': r_score,
                              'test f_score': f_score}
                    output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt")
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Test Eval results epoch*****")
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                else:
                    eval_accuracy = eval_accuracy / nb_eval_examples
                    loss = tr_loss / nb_tr_steps if args.do_train else None
                    result = {'test eval_loss': eval_loss,
                              'test eval_accuracy': eval_accuracy,
                              'test global_step': global_step,
                              'test loss': loss}
                    output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt.final")
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Eval results *****")
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
    # Save a trained model


    # Load a trained model that you have fine-tuned
    # if args.do_eval or args.do_test:
    #     model_state_dict = torch.load(output_model_file)
    #     model = BertForSequenceDisfluency.from_pretrained(args.bert_model, state_dict=model_state_dict,
    #                                                       num_labels=num_labels, num_labels_tagging=num_labels_tagging)
    #     model.to(device)
    #
    # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     eval_examples = processor.get_dev_examples(args.data_dir)
    #     eval_features = convert_examples_to_features(
    #         eval_examples, label_list, label_disf_list, args.max_seq_length, tokenizer, args.sel_prob, "dev")
    #     logger.info("***** Running evaluation on dev *****")
    #     logger.info("  Num examples = %d", len(eval_examples))
    #     logger.info("  Batch size = %d", args.eval_batch_size)
    #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    #     all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
    #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_disf_ids)
    #     # Run prediction for full data
    #     eval_sampler = SequentialSampler(eval_data)
    #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #
    #     model.eval()
    #     eval_loss, eval_accuracy = 0, 0
    #     nb_eval_steps, nb_eval_examples = 0, 0
    #
    #     # predict_result_pair = []
    #     predict_result_tagging = []
    #     # gold_result_pair = []
    #     gold_result_tagging = []
    #     input_mask_tagging = []
    #
    #     for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader, desc="Evaluating"):
    #         input_ids = input_ids.to(device)
    #         input_mask = input_mask.to(device)
    #         segment_ids = segment_ids.to(device)
    #         label_ids = label_ids.to(device)
    #         label_disf_ids = label_disf_ids.to(device)
    #
    #         with torch.no_grad():
    #             tmp_eval_loss = model(input_ids=input_ids,
    #                                   token_type_ids=segment_ids,
    #                                   attention_mask=input_mask,
    #                                   labels_tagging=label_disf_ids)
    #             logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
    #                                                 attention_mask=input_mask)
    #
    #         logits_pair = logits_pair.detach().cpu().numpy()
    #         logits_tagging = logits_tagging.detach().cpu().numpy()
    #         label_ids = label_ids.to('cpu').numpy()
    #         label_disf_ids = label_disf_ids.to('cpu').numpy()
    #         input_mask = input_mask.to('cpu').numpy()
    #
    #         if args.do_tagging:
    #             predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
    #             gold_result_tagging.append(label_disf_ids.tolist())
    #             input_mask_tagging.append(input_mask.tolist())
    #         else:
    #             tmp_eval_accuracy = accuracy(logits_pair, label_ids)
    #             # print (np.argmax(logits_pair, axis=1))
    #             # print (logits_pair)
    #             # print (label_ids)
    #             eval_accuracy += tmp_eval_accuracy
    #
    #         eval_loss += tmp_eval_loss.mean().item()
    #         nb_eval_examples += input_ids.size(0)
    #         nb_eval_steps += 1
    #     eval_loss = eval_loss / nb_eval_steps
    #     if args.do_tagging:
    #         p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging, gold_result_tagging,
    #                                                      input_mask_tagging,
    #                                                      os.path.join(args.output_dir, "dev_results.txt.finall"))
    #         result = {'eval_loss': eval_loss,
    #                   'dev p_score': p_score,
    #                   'dev r_score': r_score,
    #                   'dev f_score': f_score}
    #
    #         output_eval_file = os.path.join(args.output_dir, "dev_eval_results.txt.finall")
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Dev Eval results *****")
    #             for key in sorted(result.keys()):
    #                 logger.info("  %s = %s", key, str(result[key]))
    #                 writer.write("%s = %s\n" % (key, str(result[key])))
    #
    #
    #                 # for key in sorted(result.keys()):
    #                 #     logger.info("  %s = %s", key, str(result[key]))
    #     else:
    #         eval_accuracy = eval_accuracy / nb_eval_examples
    #         loss = tr_loss / nb_tr_steps if args.do_train else None
    #         result = {'dev eval_loss': eval_loss,
    #                   'dev eval_accuracy': eval_accuracy,
    #                   'dev global_step': global_step,
    #                   'dev loss': loss}
    #         output_eval_file = os.path.join(args.output_dir, "dev_eval_results.txt.finall")
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results *****")
    #             for key in sorted(result.keys()):
    #                 logger.info("  %s = %s", key, str(result[key]))
    #                 writer.write("%s = %s\n" % (key, str(result[key])))
    #
    # if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     eval_examples = processor.get_test_examples(args.data_dir)
    #     eval_features = convert_examples_to_features(
    #         eval_examples, label_list, label_disf_list, args.max_seq_length, tokenizer, args.sel_prob, "test")
    #     logger.info("***** Running evaluation on test *****")
    #     logger.info("  Test Num examples = %d", len(eval_examples))
    #     logger.info("  Test Batch size = %d", args.eval_batch_size)
    #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    #     all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
    #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_disf_ids)
    #     # Run prediction for full data
    #     eval_sampler = SequentialSampler(eval_data)
    #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #
    #     model.eval()
    #     eval_loss, eval_accuracy = 0, 0
    #     nb_eval_steps, nb_eval_examples = 0, 0
    #
    #     # predict_result_pair = []
    #     predict_result_tagging = []
    #     # gold_result_pair = []
    #     gold_result_tagging = []
    #     input_mask_tagging = []
    #
    #     for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader, desc="Evaluating"):
    #         input_ids = input_ids.to(device)
    #         input_mask = input_mask.to(device)
    #         segment_ids = segment_ids.to(device)
    #         label_ids = label_ids.to(device)
    #         label_disf_ids = label_disf_ids.to(device)
    #
    #         with torch.no_grad():
    #             tmp_eval_loss = model(input_ids=input_ids,
    #                                   token_type_ids=segment_ids,
    #                                   attention_mask=input_mask,
    #                                   labels_tagging=label_disf_ids)
    #             logits_pair, logits_tagging = model(input_ids=input_ids, token_type_ids=segment_ids,
    #                                                 attention_mask=input_mask)
    #         logits_pair = logits_pair.detach().cpu().numpy()
    #         logits_tagging = logits_tagging.detach().cpu().numpy()
    #         label_ids = label_ids.to('cpu').numpy()
    #         label_disf_ids = label_disf_ids.to('cpu').numpy()
    #         input_mask = input_mask.to('cpu').numpy()
    #
    #         if args.do_tagging:
    #             predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
    #             gold_result_tagging.append(label_disf_ids.tolist())
    #             input_mask_tagging.append(input_mask.tolist())
    #         else:
    #             tmp_eval_accuracy = accuracy(logits_pair, label_ids)
    #             # print (np.argmax(logits_pair, axis=1))
    #             # print (logits_pair)
    #             # print (label_ids)
    #             eval_accuracy += tmp_eval_accuracy
    #
    #         eval_loss += tmp_eval_loss.mean().item()
    #         nb_eval_examples += input_ids.size(0)
    #         nb_eval_steps += 1
    #     eval_loss = eval_loss / nb_eval_steps
    #     if args.do_tagging:
    #         p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging, gold_result_tagging,
    #                                                      input_mask_tagging,
    #                                                      os.path.join(args.output_dir, "test_results.txt.finall"))
    #         result = {'test_loss': eval_loss,
    #                   'test p_score': p_score,
    #                   'test r_score': r_score,
    #                   'test f_score': f_score}
    #         output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt.finall")
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Test Eval results *****")
    #             for key in sorted(result.keys()):
    #                 logger.info("  %s = %s", key, str(result[key]))
    #                 writer.write("%s = %s\n" % (key, str(result[key])))
    #     else:
    #         eval_accuracy = eval_accuracy / nb_eval_examples
    #         loss = tr_loss / nb_tr_steps if args.do_train else None
    #         result = {'test eval_loss': eval_loss,
    #                   'test eval_accuracy': eval_accuracy,
    #                   'test global_step': global_step,
    #                   'test loss': loss}
    #         output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt.finall")
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results *****")
    #             for key in sorted(result.keys()):
    #                 logger.info("  %s = %s", key, str(result[key]))
    #                 writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
