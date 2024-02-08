from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, Dataset

from transformers import (BertConfig,
                          BertForTokenClassification,
                          BertTokenizer)
import torch
import pandas as pd
from tqdm import tqdm
import string

import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter
import ast
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, Linear, to_hetero
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from pyg_until import SequenceEncoder, GenresEncoder, load_node_csv, load_edge_csv, node_range
from datasets import Dataset as DT
import pickle 

# remove stopwords, keep the negations
keeped_words = ['without', 'within', 'w/o', 'against', 'no', 'not', 'nt', "n't", 'against', "aren't", 'but',
                "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
                "needn't", 'no', 'nor', 'not', 'now', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                "won't", 'wouldn', "wouldn't", 'after', 'again', 'aren', 'before', 'couldn', 'didn', 'doesn', 'don',
                'hadn', 'hasn', 'haven', 'isn', 'mightn', 'mustn', "mustn't", 'needn', 'shouldn']
stop_words = set([w for w in stopwords.words('english') if w not in keeped_words] + ["'s", "s'", "please",'refills:*0', 'po','sig','bid', 'tablet', 'mg'])

exclude_dep_type = ['ROOT','punct','quantmod','mwe','dative', 'amod@nmod', 'meta', 'preconj', 'cop','pobj','det:predet']
exclude_pos_type = ['PUNCT']

def exclude_words(x):
  x=x.strip()
  if x in stop_words or x.isdigit() or x in string.punctuation:
    return True
  else:
    return False

def process_conpound(data):
  # data['new_sentence_attribute'] = None
  data['new_mt'] = None
  data['new_dependency'] = None
  data['mt_token_indice'] = None
  data['start2mt_indices'] = None
  #print('exclude_pos_type',exclude_pos_type)
  for d_index,row in data.iterrows():

    sentence_attribute = ast.literal_eval(row['sentence_attribute'])

    new_mt = []
    new_dep = []
    new_mt_tok = []
    start2mt_indices = {}
    for sent_info in sentence_attribute:
      # all_mts = ast.literal_eval(sent_info[3])
      all_mts = sent_info[3]
      new_dependencies = {}
      mt_indices2start = {}
      sent_start = sent_info[1]
      mt_indices = []
      if sent_info[4] != None:
        all_dependencies = sent_info[4]
        mt_indices = sent_info[5]
        for k, mt in enumerate(all_mts):
          start2mt_indices[mt[2][0]] = [j for j in mt[2]]
          for mt_i in range(mt[2][0], mt[2][1]):
            mt_indices2start[mt_i] = (mt[2][0], mt[0])
        for index, dep in all_dependencies.items():
          index = index + sent_start
          if index not in mt_indices:
            temp = all_dependencies[index - sent_start]
            new_temp = []
            for d in temp:
              if d[0] in exclude_dep_type or d[5] in exclude_pos_type or d[6] in exclude_pos_type or exclude_words(d[1]) or exclude_words(d[2]):
                continue
              start = d[3] + sent_start
              end = d[4] + sent_start
              if start == end:
                    continue
              a = mt_indices2start[start][0] if start in mt_indices2start else start
              b = mt_indices2start[end][0] if end in mt_indices2start else end
              if a == b: continue
              a_word = mt_indices2start[start][1] if start in mt_indices2start else d[1]
              b_word = mt_indices2start[end][1] if end in mt_indices2start else d[2]
              if a != start:
                a_pos = 'COMPOUND'
              else:
                a_pos = d[5]
              if b != end:
                b_pos = 'COMPOUND'
              else:
                b_pos = d[6]
              new_temp.append([d[0], a_word, b_word, a, b, a_pos, b_pos])
            new_dependencies[index] = new_temp
          else:
            if index in start2mt_indices:
              temp = []
              cur_indices = start2mt_indices[index]
              cur_indices_list = list(range(cur_indices[0], cur_indices[1]))
              for idx in cur_indices_list:
                cur_dep = all_dependencies[idx - sent_start]
                for dep in cur_dep:
                  if dep[0] in exclude_dep_type or dep[5] in exclude_pos_type or dep[6] in exclude_pos_type or exclude_words(dep[1]) or exclude_words(dep[2]):
                    continue
                  start = dep[3] + sent_start
                  end = dep[4] + sent_start
                  if start == end:
                    continue
                  if start in cur_indices_list and dep[4] in cur_indices_list:
                    continue
                  a = mt_indices2start[start][0] if start in mt_indices2start else start
                  b = mt_indices2start[end][0] if end in mt_indices2start else end
                  if a == b: continue
                  a_word = mt_indices2start[start][1] if start in mt_indices2start else dep[1]
                  b_word = mt_indices2start[end][1] if end in mt_indices2start else dep[2]
                  if a != start:
                    a_pos = 'COMPOUND'
                  else:
                    a_pos = dep[5]
                  if b != end:
                    b_pos = 'COMPOUND'
                  else:
                    b_pos = dep[6]
                  if [dep[0], dep[1], dep[2], a, b, dep[5], dep[6]] not in temp:
                    temp.append([dep[0], a_word, b_word, a, b, a_pos, b_pos])
              new_dependencies[index] = temp
        for d_i, d_d in new_dependencies.items():
          for each_d in d_d:
            if each_d not in new_dependencies[each_d[4]]:
              new_dependencies[each_d[4]].append(each_d)
      sent_info.append(new_dependencies)
      sent_info.append(start2mt_indices)
      sent_info.append(mt_indices)
      new_mt.append(sent_info[3])
      new_dep.append(sent_info[6])
      new_mt_tok.append(sent_info[8])
    data.at[d_index, 'new_mt'] = new_mt
    data.at[d_index, 'new_dependency'] = json.dumps(new_dep)
    data.at[d_index, 'mt_token_indice'] = new_mt_tok
    data.at[d_index, 'start2mt_indices'] = {str(k):v for k,v in start2mt_indices.items()}

  return data

def get_dep_pos_types(args,data,train_or_test):
  record_dep_types = defaultdict(list)
  pos_types = []
  for index,row in data.iterrows():
    all_sent_dep = json.loads(row['new_dependency'])
    for sent_info in all_sent_dep:
      for _,dep in sent_info.items():
        for d in dep:
          if d[0] in ['ROOT','punct']:
            continue
          record_dep_types[d[0]].append('|'.join(sorted((d[1],d[2]))))
          if d[-2] not in pos_types:
            pos_types.append(d[-2])
          if d[-1] not in pos_types:
            pos_types.append(d[-1])

  record_dep_types = {k:Counter(v) for k,v in record_dep_types.items()}
  record_dep_types = {k:dict(sorted(v.items(), key=lambda item: item[1],reverse=True)[:20]) for k,v in record_dep_types.items()}

  # with open(args.output_dir+train_or_test+'_dep_types.pkl', 'wb') as f:
  #   pickle.dump(record_dep_types, f)
  dep_types = list(record_dep_types.keys())
  return pos_types, dep_types

def get_edge_type(row, mt_l, mod_l):
  if row['sources'] in mt_l and row['dest'] in mt_l:
    return 0 # mt-mt
  elif row['sources'] in mt_l and row['dest'] in mod_l:
    return 1 # mt-mod
  elif row['sources'] in mod_l and row['dest'] in mt_l:
    return 2 # mt-mod
  else:
    return 3 # mod-mod

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, cmap="Set2")
    plt.show()

def find_adj_nodes(node, dependencies, start2mt_indices, idx_mapping, max_seq_length, skip = [], directed = False):
  s_word = []
  d_word = []
  sources = []
  dest = []
  sources_pos = []
  dest_pos = []
  dep_type = []

  s_idx = []
  d_idx = []
  mt_info =[]
  for adj_dep in dependencies[str(node)]:
    if adj_dep[3] not in idx_mapping or adj_dep[4] not in idx_mapping:
      continue
    if str(adj_dep[3]) in start2mt_indices:
      if start2mt_indices[str(adj_dep[3])][1] - 1 not in idx_mapping:
        continue
    if str(adj_dep[4]) in start2mt_indices:
      if start2mt_indices[str(adj_dep[4])][1] - 1 not in idx_mapping:
        continue
    if str(adj_dep[3]) in skip or str(adj_dep[4]) in skip:
      continue

    if directed:
      if str(adj_dep[3]) == node:
        d_word.append(adj_dep[2])
        dest.append(str(adj_dep[4]))
        # dest.append(str(adj_dep[3]))
        dest_pos.append(adj_dep[6])
        # dest_pos.append(adj_dep[5])
        dep_type.append(adj_dep[0])
        # if str(adj_dep[3]) in start2mt_indices:
        #   d_idx.append(start2mt_indices[str(adj_dep[3])])
        # else:
        #   d_idx.append((adj_dep[3], adj_dep[3]+1))
        if str(adj_dep[4]) in start2mt_indices:
          # idx_mapping
          # s_idx.append((idx_mapping[start2mt_indices[str(adj_dep[4])][0]][0],idx_mapping[start2mt_indices[str(adj_dep[4])][1]][1]))
          d_idx.append(start2mt_indices[str(adj_dep[4])])

        else:
          d_idx.append((adj_dep[4], adj_dep[4]+1))
    else:

      s_word.append(adj_dep[1])
      d_word.append(adj_dep[2])

      sources.append(str(adj_dep[3]))
      dest.append(str(adj_dep[4]))

      if len(adj_dep[1].split())>1:
        s_pos = 'COMPOUND'
      else:
        s_pos = adj_dep[5]
      if len(adj_dep[2].split())>1:
        d_pos = 'COMPOUND'
      else:
        d_pos = adj_dep[6]
      sources_pos.append(s_pos)
      dest_pos.append(d_pos)
      dep_type.append(adj_dep[0])

      if str(adj_dep[3]) in start2mt_indices:
        s_idx.append(start2mt_indices[str(adj_dep[3])])
      else:
        s_idx.append((adj_dep[3], adj_dep[3]+1))
      if str(adj_dep[4]) in start2mt_indices:
        d_idx.append(start2mt_indices[str(adj_dep[4])])
      else:
        d_idx.append((adj_dep[4], adj_dep[4]+1))

      if str(adj_dep[3]) == node:
        mt_info = [adj_dep[1], str(adj_dep[3]), s_pos, s_idx]
      elif str(adj_dep[4]) == node:
        mt_info = [adj_dep[2], str(adj_dep[4]), d_pos, d_idx]

      # s_word.extend([adj_dep[1], adj_dep[2]])
      # d_word.extend([adj_dep[2], adj_dep[1]])

      # sources.extend([adj_dep[3], adj_dep[4]])
      # dest.extend([adj_dep[4], adj_dep[3]])
      # sources_pos.extend([adj_dep[5], adj_dep[6]])
      # dest_pos.extend([adj_dep[6], adj_dep[5]])
      # dep_type.extend([adj_dep[0],adj_dep[0]])
  if directed:
    return d_word, dest, dest_pos, dep_type, d_idx
  else:
    return s_word, d_word, sources, dest, sources_pos, dest_pos, dep_type, s_idx, d_idx, mt_info

class MIMIC_Depparsed_Dataset(Dataset):
  def __init__(self, data, dep_types, pos_types, args, train_or_test):
    self.data = data
    self.args = args
    self.dep_types = dep_types
    self.pos_types = pos_types
    self.formatted_features = []
    self.train_or_test = train_or_test
    self.all_exclude = ['mm','cc']

    self.mt_per_chunk = []
    self.mod_per_mt = []
    self.graph_per_note = []
    self.nodes_per_graph = []
    self.edges_per_graph = []

    if self.args.load_graph:
      if train_or_test == 'train':
        self.graphs = torch.load(self.args.graph_dir + 'train_graph.pt')
      elif train_or_test == 'test':
        self.graphs = torch.load(self.args.graph_dir + 'test_graph.pt')
    else:
      self.graphs = []
    self.dataset = self.process_examples()
    if not self.args.load_graph:
      torch.save(self.graphs, self.args.output_dir + train_or_test +'_graph.pt')

  def map_index_from_raw_2_tokenized(self, doc):
    idx_mapping = {}
    tokenized_words = []
    doc_tokens =self.args.tokenizer.tokenize(doc.text) 
    for sent_i, sent in enumerate(doc.sents): 
      for w_i, word in enumerate(sent):
        word_t = word.text
        word_tokens = self.args.tokenizer.tokenize(word_t)
        if len(word_tokens) == 0:
          continue
        if len(tokenized_words) + len(word_tokens)+1 > self.args.max_seq_length - 2:
          break
        idx_mapping[word.i] = (len(tokenized_words)+1, len(tokenized_words) + len(word_tokens)+1)
        tokenized_words.extend(word_tokens)
      if len(tokenized_words) + len(word_tokens)+1 > self.args.max_seq_length - 2:
        break
    return idx_mapping

  def encode(self,examples):

    tokenized_note = self.args.tokenizer(examples["TEXT"], padding="max_length",max_length=self.args.max_seq_length, truncation=True)
    return tokenized_note

  def __len__(self):
        return len(self.formatted_features)
  def get_all_items(self):
    out = []
    # if self.args.pure_bert:
    #   out.append([torch.tensor(data['ID']), torch.tensor(data['input_ids']), torch.tensor(data['token_type_ids']), torch.tensor(data['attention_mask']), torch.tensor(data['Label'])])
    for idx in tqdm(range(len(self.dataset)),desc='get all items'):
        data = self.dataset[idx]
        if not self.args.pure_bert:
          graph = self.graphs[idx]
          out.append([torch.tensor(data['ID']), torch.tensor(data['input_ids']), torch.tensor(data['token_type_ids']), torch.tensor(data['attention_mask']), graph, torch.tensor(data['Label'])])
        else:
          out.append([torch.tensor(data['ID']), torch.tensor(data['input_ids']), torch.tensor(data['token_type_ids']), torch.tensor(data['attention_mask']), torch.tensor(data['Label'])])
    return out

  def get_range(self, sentence, start, end):
        pre = sentence[:start]
        tokenized_pre = self.args.tokenizer.tokenize(pre)
        word = sentence[start:end]
        tokenized_word = self.args.tokenizer.tokenize(word)
        return torch.tensor([len(tokenized_pre), len(tokenized_pre) + len(tokenized_word)])
  def get_statistics(self):
    # self.mt_per_chunk = []
    # self.mod_per_mt = []
    # self.graph_per_note = []
    # self.nodes_per_graph = []
    # self.edges_per_graph = []
    print(self.train_or_test)
    # print('Ave # of MT per chunk:',sum(self.mt_per_chunk)/len(self.mt_per_chunk))
    # print('Ave # of modifier per MT:',sum(self.mod_per_mt)/len(self.mod_per_mt))
    # print('Ave # of graph per note:',sum(self.graph_per_note)/len(self.graph_per_note))
    # print('Ave # of nodes per graph:',sum(self.nodes_per_graph)/len(self.nodes_per_graph))
    # print('Ave # of edges per graph:',sum(self.edges_per_graph)/len(self.edges_per_graph))

    print('# of MT per chunk: min, max, mean:',min(self.mt_per_chunk),max(self.mt_per_chunk),sum(self.mt_per_chunk)/len(self.mt_per_chunk))
    print('# of modifier per MT: min, max, mean:',min(self.mod_per_mt), max(self.mod_per_mt), sum(self.mod_per_mt)/len(self.mod_per_mt))
    print('# of graph per note: min, max, mean:',min(self.graph_per_note), max(self.graph_per_note), sum(self.graph_per_note)/len(self.graph_per_note))
    print('# of nodes per graph: min, max, mean:',min(self.nodes_per_graph), max(self.nodes_per_graph), sum(self.nodes_per_graph)/len(self.nodes_per_graph))
    print('# of edges per graph: min, max, mean:',min(self.edges_per_graph), max(self.edges_per_graph), sum(self.edges_per_graph)/len(self.edges_per_graph))
  def build_empty_graph(self):
    data = HeteroData()
    data['mt'].x = torch.empty(0, len(self.pos_types)+2, dtype=torch.long)
    data['mod'].x = torch.empty(0, len(self.pos_types)+2, dtype=torch.long)

    data['mt', 'dep', 'mt'].edge_index = torch.empty(2, 0, dtype=torch.long)
    data['mt', 'dep', 'mt'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)
    data['mt', 'dep', 'mod'].edge_index = torch.empty(2, 0, dtype=torch.long)
    data['mt', 'dep', 'mod'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)
    data['mod', 'dep', 'mt'].edge_index = torch.empty(2, 0, dtype=torch.long)
    data['mod', 'dep', 'mt'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)
    data['mod', 'dep', 'mod'].edge_index = torch.empty(2, 0, dtype=torch.long)
    data['mod', 'dep', 'mod'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)
    return data


  def construct_graph(self, note_mts, note_dependencies, note_mt_token_indices, nlp, start2mt_indices, idx_mapping):

    note_edge_df = pd.DataFrame()
    note_node_df = pd.DataFrame()
    graph_count = 0
    mt_count = 0

    for sent_i in range(len(note_mts)):
      
      # sent_graphs = []
      all_mts = note_mts[sent_i]
      mt_count += len(all_mts)
      
      all_dependencies = note_dependencies[sent_i]
      mt_token_indices = note_mt_token_indices[sent_i]
      mt_token_indices = [str(k) for k in mt_token_indices]

      mt_seen = []
      mt_wo_dep = []

      i = 0
      j = 0
      sent_s_word = []
      sent_d_word = []
      sent_source = []
      sent_dest = []
      sent_source_pos = []
      sent_dest_pos = []
      sent_dep_type = []

      sent_s_idx = []
      sent_d_idx = []
      
      while len(set(mt_seen)) < len(all_mts):
        # node_count = 0
        # edge_count = 0

        if all_mts[j][2][0] in mt_seen:
          j+=1
          continue
        cur_mt_info = all_mts[j]
        cur_mt = cur_mt_info[0]
        cui = cur_mt_info[4]
        tui = cur_mt_info[5]
        mt_idx = str(cur_mt_info[2][0])
        if cur_mt in self.all_exclude:
          j+=1
          mt_seen.append(mt_idx)
          continue
        if int(mt_idx) not in idx_mapping:
          mt_seen.append(mt_idx)
          j+=1
          continue
        s_word, d_word, sources, dest, sources_pos, dest_pos, dep_type, s_idx, d_idx, mt_info = find_adj_nodes(mt_idx, all_dependencies, start2mt_indices, idx_mapping, self.args.max_seq_length)
        self.mod_per_mt.append(len(s_word))
        sent_s_word.extend(s_word)
        sent_d_word.extend(d_word)
        sent_source.extend(sources)
        sent_dest.extend(dest)
        sent_source_pos.extend(sources_pos)
        sent_dest_pos.extend(dest_pos)
        sent_dep_type.extend(dep_type)
        mt_seen.append(mt_idx)
        sent_s_idx.extend(s_idx)
        sent_d_idx.extend(d_idx)

        if len(s_word) == 0:
          if len(cur_mt.split())>1:
            _pos = 'COMPOUND'
          else:
            _pos = nlp(cur_mt)[0].pos_
          if _pos in exclude_pos_type:
            j+=1
            continue
          else:

            if int(mt_idx) in idx_mapping:
              if mt_idx in start2mt_indices:
                cur_mt_idx_range = start2mt_indices[mt_idx]
                if cur_mt_idx_range[1] - 1 in idx_mapping:
                  mt_wo_dep.append((cur_mt, mt_idx, _pos, cur_mt_idx_range))
              else:
                cur_mt_idx_range = (int(mt_idx), int(mt_idx) + 1)
                mt_wo_dep.append((cur_mt, mt_idx, _pos, cur_mt_idx_range))

        if self.args.multi_hop:
            adj_nodes = list(sent_dest)
            adj_nodes_word = list(d_word)
            # adj_nodes = list(sent_source + sent_dest)
            # adj_nodes_word = list(s_word + d_word)
            visited = []
            for adj_i,adj_n in enumerate(adj_nodes):
              adj_n = str(adj_n)
              if adj_n in visited:
                continue
              else:
                visited.append(adj_n)
              
              if adj_n != mt_idx:
                # s_word, d_word, sources, dest, sources_pos, dest_pos, dep_type, s_idx, d_idx, _ = find_adj_nodes(adj_n, all_dependencies, start2mt_indices, idx_mapping, self.args.max_seq_length, [mt_idx])
                d_word, dest, dest_pos, dep_type, d_idx = find_adj_nodes(adj_n, all_dependencies, start2mt_indices, idx_mapping, self.args.max_seq_length, [mt_idx], True)
                self.mod_per_mt.append(len(d_word))
                # s_word = [cur_mt] * len(s_word)
                # sources = [start2mt_indices[mt_idx]] * len(sources)
                # if len(cur_mt.split())>1:
                #   _pos = 'COMPOUND'
                # else:
                #   _pos = nlp(cur_mt)[0].pos_
                # sources_pos = [_pos] * len(sources_pos)
                # mt_info
                if len(d_word) > 0:
                  s_word = [mt_info[0]] * len(d_word)
                  sources = [mt_info[1]] * len(d_word)
                  sources_pos = [mt_info[2]] * len(d_word)
                  s_idx = [mt_info[3]] * len(d_word)
                  sent_s_word.extend(s_word)
                  sent_d_word.extend(d_word)
                  sent_source.extend(sources)
                  sent_dest.extend(dest)
                  sent_source_pos.extend(sources_pos)
                  sent_dest_pos.extend(dest_pos)
                  sent_dep_type.extend(dep_type)

                  sent_s_idx.extend(s_idx)
                  sent_d_idx.extend(d_idx)
                if adj_n in mt_token_indices:
                  if len(s_word) > 0:
                    if adj_n not in mt_seen:
                      mt_adj_nodes = list(sources + dest)
                      mt_adj_nodes_word = list(s_word + d_word)
                      mt_visited = []
                      for mt_adj_i,mt_adj_n in enumerate(mt_adj_nodes):
                        mt_adj_n = str(mt_adj_n)
                        if mt_adj_n in mt_visited:
                          continue
                        else:
                          mt_visited.append(mt_adj_n)
                        if mt_adj_n != adj_n:
                          mt_s_word, mt_d_word, mt_sources, mt_dest, mt_sources_pos, mt_dest_pos, mt_dep_type, mt_s_idx, mt_d_idx, _ = find_adj_nodes(mt_adj_n, all_dependencies, start2mt_indices, idx_mapping, self.args.max_seq_length, [adj_n])
                          self.mod_per_mt.append(len(mt_s_word))
                          sent_s_word.extend(mt_s_word)
                          sent_d_word.extend(mt_d_word)
                          sent_source.extend(mt_sources)
                          sent_dest.extend(mt_dest)
                          sent_source_pos.extend(mt_sources_pos)
                          sent_dest_pos.extend(mt_dest_pos)
                          sent_dep_type.extend(mt_dep_type)

                          sent_s_idx.extend(mt_s_idx)
                          sent_d_idx.extend(mt_d_idx)
                  mt_seen.append(adj_n)
        j += 1
        graph_count += 1
        # sent_source = [str(int(node_id)+sent_start[sent_i]) for node_id in sent_source]
        # sent_dest = [str(int(node_id)+sent_start[sent_i]) for node_id in sent_dest]
        
        df = pd.DataFrame({'s_word': sent_s_word, 'd_word': sent_d_word, 'sources': sent_source, 'dest': sent_dest, \
                        'sources_pos': sent_source_pos, 'dest_pos': sent_dest_pos, 'dep_type': sent_dep_type})
        df = df.drop_duplicates()
        _temp_pos = sent_source_pos + sent_dest_pos
        node_df = pd.DataFrame({'word': sent_s_word + sent_d_word, 'id': sent_source + sent_dest, 'pos': sent_source_pos + sent_dest_pos, 'word_type': [1  if (x in mt_token_indices and _temp_pos[_i] != 'ADJ') else 0 for _i,x in enumerate(sent_source + sent_dest)], 'node_idx': sent_s_idx + sent_d_idx})

        if len(mt_wo_dep) > 0:
          mt_wo_dep_df = pd.DataFrame({'word': [p[0] for p in mt_wo_dep], 'id': [p[1] for p in mt_wo_dep], 'pos': [p[2] for p in mt_wo_dep],'word_type': [1 if p[2]!='ADJ' else 0 for p in mt_wo_dep], 'node_idx':[p[3] for p in mt_wo_dep]})
          node_df = pd.concat([node_df, mt_wo_dep_df])

        self.nodes_per_graph.append(len(node_df))
        self.edges_per_graph.append(len(df))

        note_edge_df = pd.concat([note_edge_df, df])
        note_node_df = pd.concat([note_node_df, node_df])
        note_node_df = note_node_df.drop_duplicates('id')
        note_edge_df = note_edge_df.drop_duplicates(['sources','dest'])
        sent_s_word = []
        sent_d_word = []
        sent_source = []
        sent_dest = []
        sent_source_pos = []
        sent_dest_pos = []
        sent_dep_type = []
        sent_s_idx = []
        sent_d_idx = []
    self.graph_per_note.append(graph_count)
    self.mt_per_chunk.append(mt_count)
    node_df = note_node_df
    node_df = node_df.drop_duplicates('id')
    df = note_edge_df
    df = df.reset_index(drop=True)
    node_df = node_df.reset_index(drop=True)

    if len(node_df) > 0:

      mt_node_df = node_df[node_df['word_type'] == 1]
      if len(mt_node_df) > 0:
        mod_node_df = node_df[node_df['word_type'] == 0]
        mod_l = mod_node_df['id'].values.tolist()
        mt_l = mt_node_df['id'].values.tolist()

        df['edge_type'] = None

        for e_index,row in df.iterrows():
          e_type = get_edge_type(row, mt_l, mod_l)
          df.at[e_index, 'edge_type'] = e_type

        mt_mt_df = df[df['edge_type'] == 0].drop_duplicates()
        mt_mod_df = df[df['edge_type'] == 1].drop_duplicates()
        mod_mt_df = df[df['edge_type'] == 2].drop_duplicates()
        # mt_mod_df['mt'] = mt_mod_df.apply(lambda x: x['sources'] if x['sources'] in mt_l else x['dest'], axis=1)
        # mt_mod_df['mod'] = mt_mod_df.apply(lambda x: x['sources'] if x['sources'] in mod_l else x['dest'], axis=1)
        # for index,row in
        mod_mod_df = df[df['edge_type'] == 3].drop_duplicates()

        mt_mt_df = mt_mt_df.drop_duplicates()
        mt_mod_df = mt_mod_df.drop_duplicates()
        mod_mt_df = mod_mt_df.drop_duplicates()
        mod_mod_df = mod_mod_df.drop_duplicates()

        data = HeteroData()

        mt_x, mt_mapping = load_node_csv(
        mt_node_df, index_col='id', map_start=0, encoders={
            # 'word': SequenceEncoder(model_name = self.args.bert_model_dir),
            'pos': GenresEncoder(self.pos_types),
            'node_idx': node_range(idx_mapping)
        })

        #_, user_mapping = load_node_csv(mt_node_df, index_col='userId')


        map_start = len(mt_mapping.items())

        if len(mod_node_df) > 0:

          mod_x, mod_mapping = load_node_csv(
          mod_node_df, index_col='id', map_start=0, encoders={
              # 'word': SequenceEncoder(model_name = self.args.bert_model_dir),
              'pos': GenresEncoder(self.pos_types),
              'node_idx': node_range(idx_mapping)
          })
          data['mod'].x = mod_x
        else:
          data['mod'].x = torch.empty(0, len(self.pos_types)+2, dtype=torch.long)

        data['mt'].x = mt_x

        if len(mt_mt_df)>0:

          mt_mt_index, mt_mt_label = load_edge_csv(
              mt_mt_df,
              src_index_col='sources',
              src_mapping=mt_mapping,
              dst_index_col='dest',
              dst_mapping=mt_mapping,
              encoders={'dep_type': GenresEncoder(self.dep_types)},
          )
          data['mt', 'dep', 'mt'].edge_index = mt_mt_index
          data['mt', 'dep', 'mt'].edge_label = mt_mt_label
        else:
          data['mt', 'dep', 'mt'].edge_index = torch.empty(2, 0, dtype=torch.long)
          data['mt', 'dep', 'mt'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)

        if len(mt_mod_df) > 0:

          mt_mod_index, mt_mod_label = load_edge_csv(
              mt_mod_df,
              src_index_col='sources',
              src_mapping=mt_mapping,
              dst_index_col='dest',
              dst_mapping=mod_mapping,
              encoders={'dep_type': GenresEncoder(self.dep_types)},
          )
          data['mt', 'dep', 'mod'].edge_index = mt_mod_index
          data['mt', 'dep', 'mod'].edge_label = mt_mod_label
        else:
          data['mt', 'dep', 'mod'].edge_index = torch.empty(2, 0, dtype=torch.long)
          data['mt', 'dep', 'mod'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)

        if len(mod_mt_df) > 0:

          mod_mt_index, mod_mt_label = load_edge_csv(
              mod_mt_df,
              src_index_col='sources',
              src_mapping=mod_mapping,
              dst_index_col='dest',
              dst_mapping=mt_mapping,
              encoders={'dep_type': GenresEncoder(self.dep_types)},
          )
          data['mod', 'dep', 'mt'].edge_index = mod_mt_index
          data['mod', 'dep', 'mt'].edge_label = mod_mt_label
        else:
          data['mod', 'dep', 'mt'].edge_index = torch.empty(2, 0, dtype=torch.long)
          data['mod', 'dep', 'mt'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)

        if len(mod_mod_df) > 0:
          mod_mod_index, mod_mod_label = load_edge_csv(
              mod_mod_df,
              src_index_col='sources',
              src_mapping=mod_mapping,
              dst_index_col='dest',
              dst_mapping=mod_mapping,
              encoders={'dep_type': GenresEncoder(self.dep_types)},
          )
          data['mod', 'dep', 'mod'].edge_index = mod_mod_index
          data['mod', 'dep', 'mod'].edge_label = mod_mod_label
        else:
          data['mod', 'dep', 'mod'].edge_index = torch.empty(2, 0, dtype=torch.long)
          data['mod', 'dep', 'mod'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)
        return data
      else:
        return self.build_empty_graph()
    else:
      return self.build_empty_graph()

  def process_examples(self):
    dataset = DT.from_pandas(self.data)
    dataset = dataset.map(self.encode, batched=True)

    if self.args.pure_bert:
      return dataset

    all_id = []
    all_graphs = []
    all_input_ids = []
    all_seg_ids = []
    all_mask_ids = []

    count = 0
    
    nlp = spacy.load("en_core_sci_sm")
    # patiants = len(set(data['ID'].values.tolist()))
    p_chunks = self.data[self.data['Label']==1]['ID']
    n_chunks = self.data[self.data['Label']==0]['ID']
    print(self.train_or_test)

    print('# of patients:',len(set(self.data['ID'].values.tolist())))
    print('# of chunks:', len(self.data))
    print('# of positive chunks', len(p_chunks))
    print('# of negative chunks', len(n_chunks))
    print('# of positive patients:',len(set(p_chunks.values.tolist())))
    print('# of negative patients:',len(set(n_chunks.values.tolist())))

    for index,row in tqdm(self.data.iterrows(),desc='get graph'):
      sent_attr = ast.literal_eval(row['sentence_attribute'])
      # sent_start = [sent_info[1] for sent_info in sent_attr]
      note_mts = ast.literal_eval(row['new_mt'])

      note_dependencies = ast.literal_eval(row['new_dependency'])
      #note_mt_token_indices = ast.literal_eval(row['mt_token_indice'])
      note_mt_token_indices = row['mt_token_indice']
      start2mt_indices = row['start2mt_indices']
      idx_mapping = self.map_index_from_raw_2_tokenized(nlp(row['TEXT']))

      temp = {}

      for m_start, m_range in start2mt_indices.items():
        if int(m_range[1]) in idx_mapping:
          temp[m_start] = m_range
      start2mt_indices = temp

      if not self.args.pure_bert and not self.args.load_graph:

          graph = self.construct_graph(note_mts, note_dependencies, note_mt_token_indices, nlp, start2mt_indices, idx_mapping)
          # graph = T.ToUndirected()(graph)
          self.graphs.append(graph)
    return dataset

