from torch_geometric.data import Data
from torch_geometric.nn import GATConv, Linear, to_hetero, GCNConv,HGTConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn.pool import avg_pool_neighbor_x,max_pool_neighbor_x
from torch_geometric.utils import scatter
from torch_geometric.transforms import add_self_loops
# from torch_geometric.nn.models import AttentiveFP

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from modeling_readmission import BertForSequenceClassification,BertModel,BertConfig
# from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
from self_attention import SelfAttention
from attentionFP import AttentiveFP


class CHSLM(nn.Module):

    def __init__(self, args, tokenizer,metadata):
        super(CHSLM, self).__init__()
        self.args = args
        self.tokenizer=self.args.tokenizer

        self.bert = BertModel.from_pretrained(args.bert_model_dir)
        config=self.bert.config
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        # self.dropout_bert = nn.Dropout(args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        args.embedding_dim = config.hidden_size  # 768
        self.gat_encoder = HGT(metadata,['mt','mod'],args.embedding_dim, args.embedding_dim, num_heads=2, num_layers=2)

        self.linear_encoder = nn.Linear(len(self.args.pos_types) + args.embedding_dim,args.embedding_dim)

        # self.out_dim = config.hidden_size*2
        self.out_dim = config.hidden_size
        
        self.fc_final = nn.Linear(self.out_dim, args.num_classes)
        # comment for ablation study
        self.mutual=selfalignment(config.hidden_size)
        self.combine = nn.Linear(args.embedding_dim*2, args.embedding_dim)

    def forward(self, note_ids,full_input_ids_batch, full_segment_ids_batch, full_input_mask_batch, all_graphs):
        out=torch.empty(size=(note_ids.shape[0],self.out_dim)).to(self.args.device)
        for note_i in range(note_ids.shape[0]):
            note_emb, note_pooler = self.bert(
                    input_ids = torch.unsqueeze(full_input_ids_batch[note_i,:], 0),
                    attention_mask = torch.unsqueeze(full_input_mask_batch[note_i,:], 0),
                    token_type_ids = torch.unsqueeze(full_segment_ids_batch[note_i,:], 0),
                    output_all_encoded_layers = False
                )
            note_pooler = self.dropout_bert(note_pooler)            
            note_graphs = all_graphs[note_i]
        
            all_mts = []
            all_mods = []
            note_graphs = note_graphs.to(self.args.device)

            x = note_graphs.x_dict
            train_pos_edge_index = note_graphs.edge_index_dict
            if x['mt'].nelement() > 0:
                mt_range = x['mt'][:,-2:]
                mod_range = x['mod'][:,-2:]
                selected_note_emb=torch.squeeze(note_emb)
                mt_output = []
                mod_output = []

                for mt_i in range(mt_range.shape[0]):
                    indices = torch.tensor([idx for idx in range(int(mt_range[mt_i].cpu().detach().numpy()[0]),int(mt_range[mt_i].cpu().detach().numpy()[1]))]).to(self.args.device) #whether or not +1? index of the sentence in the note
                    mt_ids = torch.index_select(torch.unsqueeze(full_input_ids_batch[note_i,:], 0), 1, indices)
                    mt_ids = torch.index_select(full_input_ids_batch[note_i,:], -1, indices)
                    ori_sentence=self.args.tokenizer.decode(mt_ids.cpu().detach().numpy())
                    emb=torch.index_select(selected_note_emb, 0, indices) # sentence embedding extracted from the note embedding
                    emb=torch.unsqueeze(emb, 0)
                    max_emb=torch.max(emb,dim=1)[0]
                    mt_output.extend(max_emb)

                for mod_i in range(mod_range.shape[0]):
                    indices = torch.tensor([idx for idx in range(int(mod_range[mod_i].cpu().detach().numpy()[0]),int(mod_range[mod_i].cpu().detach().numpy()[1]))]).to(self.args.device) #whether or not +1? index of the sentence in the note
                    emb=torch.index_select(selected_note_emb, 0, indices) # sentence embedding extracted from the note embedding
                    emb=torch.unsqueeze(emb, 0)
                    max_emb=torch.max(emb,dim=1)[0]
                    mod_output.extend(max_emb)

                mt_output = torch.stack(mt_output).to(self.args.device)

                x['mt'] = torch.cat((mt_output, x['mt'][:,:-2]), dim = 1).to(self.args.device)

                if x['mod'].nelement() > 0:
                    mod_output = torch.stack(mod_output).to(self.args.device)
                    x['mod'] = torch.cat((mod_output, x['mod'][:,:-2]), dim = 1).to(self.args.device)
                    graph_out = self.gat_encoder(x, train_pos_edge_index)
                    mt_embed,mod_embed = graph_out
                else:
                    mt_embed = self.linear_encoder(x['mt'])
                   
                fact_out=pad_sequence(mt_embed,batch_first=True,padding_value=0)
                stacked_note=note_pooler.repeat(fact_out.shape[0],1)
                note_pooler_ori,attn_output,att=self.mutual(stacked_note.permute(1,0),fact_out.permute(1,0))

                note_pooler_ori=note_pooler_ori.mean(dim=1,keepdim=True)
                attn_output = torch.max(attn_output,dim=1,keepdim=True)[0]

                combined = torch.concat((note_pooler_ori.permute(1,0),attn_output.permute(1,0)),dim=1)
                note_pooler_ori = self.combine(combined)

                out[note_i]=torch.flatten(note_pooler_ori.permute(1,0))
            else:
                x_out=F.pad(note_pooler,(0,self.out_dim-note_pooler.shape[1]),'constant',0)
                out[note_i]=torch.flatten(x_out)
        logit = self.fc_final(out)
        m = nn.Sigmoid()
        logit=m(logit)
        logit=torch.flatten(logit)
        return logit

class selfalignment(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, bias=False):
        super(selfalignment, self).__init__()
        self.in_features = in_features
        self.dropout = nn.Dropout(0.1)
        self.linear=torch.nn.Linear(in_features, in_features,bias=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(in_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, text1):#b,s1,h;b,s2,h;b,s1;b,s2
        text=self.linear(text.permute(1,0))
        logit=torch.matmul(text.permute(1,0),text1.transpose(0,1))#b,s1,s2
        logits=torch.softmax(logit,-1)#b,s1,s2
        logits1 = torch.softmax(logit, -2)#b,s1,s2
        output = torch.matmul(logits,text1)
        output1 = torch.matmul(logits1.transpose(0,1),text.permute(1,0))
        return output+text.permute(1,0),output1+text1,logit

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        # self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning

        # self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning
        
        self.conv1 = GATConv((-1, -1), in_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin1 = Linear(-1, in_channels)
        self.lin2 = Linear(-1, out_channels)
        

    def forward(self, x, edge_index):
        # x = self.conv1(x, edge_index).relu()
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder,self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             normalize=True, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             normalize=True, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class HGT(torch.nn.Module):
    def __init__(self, metadata, node_types,hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['mt']),self.lin(x_dict['mod'])


class MIMIC_Bert_Only(nn.Module):

    def __init__(self, args):
        super(MIMIC_Bert_Only, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir)
        config=self.bert.config
        self.dropout = nn.Dropout(args.dropout)
        args.embedding_dim = config.hidden_size  # 768

        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, note_ids,full_input_ids_batch,full_segment_ids_batch, full_input_mask_batch): 
        _,note_pooler = self.bert(
                    input_ids=full_input_ids_batch,
                    attention_mask=full_input_mask_batch,
                    token_type_ids=full_segment_ids_batch,
                    output_all_encoded_layers = False
            )
        

        pooled_output = self.dropout(note_pooler)
        logit = self.classifier(pooled_output)

        m = nn.Sigmoid()
        logit=m(logit)
        logit=torch.flatten(logit)
        return logit
