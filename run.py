import pandas as pd
import numpy as np
import csv
import sys
import ast
from collections import defaultdict
import logging
import os
import random
import argparse

from datasets import load_dataset
import torch
from transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer, AutoTokenizer, AutoModel, DataCollatorWithPadding,BertModel)
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
import networkx as nx

import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter

import json
from torch_geometric.loader import DataLoader
from datasets import MIMIC_Depparsed_Dataset, process_conpound, get_dep_pos_types
from model import CHSLM, MIMIC_Bert_Only
from trainer import train_mimic, evaluate

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str,
                        default='../readmission_data/',
                        help='data folder.')
    parser.add_argument('--output_dir', type=str, default='output/mimic_1/',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--graph_dir', type=str, default='output/mimic_1/',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes.')
    parser.add_argument('--cuda_id', type=str, default='cuda',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed for initialization')
    parser.add_argument('--bert_model_dir', type=str, default='../clinicalBERT-master/model/pretraining/',
                        help='Path to pre-trained Bert model.')
    # parser.add_argument('--bert_model_dir', type=str, default='../RGAT-ABSA-master/model/baseline_clinical_BERT_1_epoch_512/',
    #                     help='Path to pre-trained Bert model.')
    # parser.add_argument('--bert_model_dir', type=str, default='../../biomarker/new_code/saved_models/biobert/',help='Path to pre-trained Bert model.')
    parser.add_argument('--pure_bert', action='store_true',
                        help='flat text, [cls] to predict.')
    parser.add_argument('--multi_hop', action='store_true',
                        help='Multi hop non connection.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='max_sequence_len')

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--load_model', action='store_true',
                        help='training')
    parser.add_argument('--load_classification_path', type=str, default='../clinicalBERT-master/model/pretraining/',
                        help='classification model path')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--load_graph', action='store_true',
                        help='training')
    parser.add_argument('--gnn', type=str, default='HGT', choices=['GAT', 'GCN','HGT','attentiveFP'])

    return parser.parse_args()

def check_args(args):
    '''
    eliminate confilct situations

    '''
    logger.info(vars(args))

def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Parse args
    args = parse_args()
    check_args(args)

    # Setup CUDA, GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    # device = torch.device('cpu')
    device = torch.device(args.cuda_id if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    args.graph_path = args.output_dir + 'graph.pt'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        if args.mode == 'train':
            for i in range(int(args.num_train_epochs)):
                os.makedirs(args.output_dir + str(i))

    if args.load_model:
        tokenizer = BertTokenizer.from_pretrained(args.load_classification_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
    args.tokenizer = tokenizer

    train_df = pd.read_csv(args.dataset_folder + 'train.csv')
    test_df = pd.read_csv(args.dataset_folder + 'test.csv')

    train_df = process_conpound(train_df)
    test_df = process_conpound(test_df)
    train_df['new_mt'] = train_df['new_mt'].astype(str)
    test_df['new_mt'] = test_df['new_mt'].astype(str)
    
    train_pos_types, train_dep_types = get_dep_pos_types(args,train_df,'train')
    test_pos_types, test_dep_types = get_dep_pos_types(args,test_df,'test')
    pos_types = list(set(train_pos_types + test_pos_types))
    dep_types = list(set(train_dep_types + test_dep_types))

    args.pos_types = pos_types
    args.dep_types = dep_types
    # print('pos_types',len(pos_types), pos_types)
    # print('dep_types',len(dep_types), dep_types)
    
    train_dataset = MIMIC_Depparsed_Dataset(train_df, dep_types, pos_types, args, 'train')
    test_dataset = MIMIC_Depparsed_Dataset(test_df, dep_types, pos_types, args, 'test')

    metadata = (['mod', 'mt'], [('mt', 'dep', 'mt'), ('mt', 'dep', 'mod'), ('mod', 'dep', 'mt'), ('mod', 'dep', 'mod')])
    # metadata = (['mod', 'mt'], [('mt', 'dep', 'mt'), ('mt', 'dep', 'mod'), ('mod', 'dep', 'mt'), ('mod', 'dep', 'mod'), ('mod', 'rev_dep', 'mt'), ('mt', 'rev_dep', 'mod')])

    if args.pure_bert:
        model = MIMIC_Bert_Only(args)
    else:
        model = CHSLM(args, tokenizer, metadata)
    if args.load_model:
        checkpoint = torch.load(args.load_classification_path+'model.pth', map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        model.to(args.device) 
        results, eval_loss = evaluate(args, train_dataset, model)
    else:
        model.to(args.device)
        _, _,  all_eval_results,best_metric = train_mimic(args, train_dataset, model, test_dataset)

if __name__ == "__main__":
    main()






