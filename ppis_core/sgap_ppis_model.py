import os
import pickle
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from .egnn import *
from .muse_attention import MUSEAttention

def _env_int(name, default):
    value = os.environ.get(name)
    if value is not None and value != "":
        return int(value)
    return default


def _env_float(name, default):
    value = os.environ.get(name)
    if value is not None and value != "":
        return float(value)
    return default


# Project
MODEL_NAME = "SGAP-PPIS"
ARCH_NAME = "SGAP-PPIS-GeoAPPNP-MSEAlign"

# Feature Path
Feature_Path = "./Feature/"
# Seed
SEED = _env_int("SEED", 100)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# model parameters
ADD_NODEFEATS = 'all'

MAP_CUTOFF = 14
DIST_NORM = 15

# INPUT_DIM
if ADD_NODEFEATS == 'all':  
    INPUT_DIM = 54 + 7 + 1
elif ADD_NODEFEATS == 'atom_feats':  
    INPUT_DIM = 54 + 7
elif ADD_NODEFEATS == 'psepose_embedding':  
    INPUT_DIM = 54 + 1
elif ADD_NODEFEATS == 'no':
    INPUT_DIM = 54
HIDDEN_DIM1 = _env_int("HIDDEN_DIM1", 128)
HIDDEN_DIM2 = _env_int("HIDDEN_DIM2", 256)
APPNP_LAYER = _env_int("APPNP_LAYER", 5)   # APPNP layers (replace EGAT layers)
EGNN_LAYER = _env_int("EGNN_LAYER", 5)
DROPOUT = _env_float("DROPOUT", 0.3)
ALPHA = _env_float("ALPHA", 0.7)
LAMBDA = _env_float("LAMBDA", 1.5)
APPNP_ALPHA = _env_float("APPNP_ALPHA", 0.1)
APPNP_ALPHA_MIN = _env_float("APPNP_ALPHA_MIN", 0.05)
APPNP_ALPHA_MAX = _env_float("APPNP_ALPHA_MAX", 0.95)

LEARNING_RATE = _env_float("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 0.0)
LR_FACTOR = _env_float("LR_FACTOR", 0.3)
LR_PATIENCE = _env_int("LR_PATIENCE", 5)
BATCH_SIZE = 1
NUM_CLASSES = 2  
NUMBER_EPOCHS = _env_int("EPOCHS", 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embedding(sequence_name):
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)

def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)

def normalize(mx):

    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def cal_edges(sequence_name, radius=MAP_CUTOFF):  # to get the index of the edges
    dist_matrix = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(int)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list

def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix, pos = map(list, zip(*samples))
    label = torch.as_tensor(np.asarray(label, dtype=np.float32))
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.as_tensor(np.asarray(adj_matrix, dtype=np.float32))
    pos = torch.cat(pos).float()
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix, pos

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True): 
      
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1        
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  

        self.gamma = gamma

    def forward(self, preds, labels):
      
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)

        preds_logsoft = F.log_softmax(preds, dim=1)

        preds_softmax = torch.exp(preds_logsoft)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))

        self.alpha = self.alpha.gather(0, labels.view(-1))

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

      
        loss = torch.mul(self.alpha, loss.t())
       

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss

class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM, psepos_path='./Feature/psepos/Train335_psepos_SC.pkl'):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist

    def __getitem__(self, index):
      
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        nodes_num = len(sequence)
    
        pos = self.residue_psepos[sequence_name]
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos).type(torch.FloatTensor)

        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features], axis=1)

        node_features = torch.from_numpy(node_features)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'atom_feats':
            res_atom_features = get_res_atom_features(sequence_name)
            res_atom_features = torch.from_numpy(res_atom_features)
            node_features = torch.cat([node_features, res_atom_features], dim=-1)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'psepose_embedding':
      
            node_features = torch.cat([node_features, torch.sqrt(
                torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

       
        radius_index_list = cal_edges(sequence_name, MAP_CUTOFF)
        edge_feat = self.cal_edge_attr(radius_index_list, pos)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)
        src, dst = radius_index_list[1], radius_index_list[0]
        G = dgl.graph((src, dst), num_nodes=nodes_num)
        G.edata['ex'] = torch.from_numpy(edge_feat).float()

        adj_matrix = load_graph(sequence_name)
       
        node_features = node_features.unsqueeze(0).float()

        return sequence_name, sequence, label, node_features, G, adj_matrix, pos

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
     
        pdist = nn.PairwiseDistance(p=2,keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)

        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list

    def add_edges_custom(self, G, radius_index_list, edge_features):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)


class EGNNBaseModule(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size):
        super(EGNNBaseModule, self).__init__()
        self.EGNNConv = EGNN(in_node_nf=in_size, hidden_nf=hidden_size, out_node_nf=out_size, in_edge_nf=edge_feat_size,residual=False,
                             n_layers=1,
                             attention=True)

    def forward(self, input, coord_feat, graph, efeats):
        hi, pos = self.EGNNConv(input, coord_feat, graph.edges(), efeats)
        output = hi+input
        return output, pos

class APPNPBaseModule(nn.Module):
    def __init__(self, alpha=APPNP_ALPHA):
        super(APPNPBaseModule, self).__init__()
        self.alpha = alpha

    def forward(self, input, h0, adj_matrix, adaptive_alpha=None):
        if adj_matrix is None:
            print('ERROR: APPNP branch needs adj_matrix.')
            raise ValueError
        if adj_matrix.dim() == 3:
            adj_matrix = torch.squeeze(adj_matrix, dim=0)
        hi = torch.mm(adj_matrix.float(), input)
        if adaptive_alpha is None:
            alpha_value = self.alpha
        else:
            alpha_value = torch.clamp(adaptive_alpha, min=APPNP_ALPHA_MIN, max=APPNP_ALPHA_MAX)
        output = (1.0 - alpha_value) * hi + alpha_value * h0
        return output


class EGAT_EGNN(nn.Module):
    def __init__(self, egat_nlayers, egnn_nlayers, nfeat, nhidden1,nhidden2, nclass, dropout, lamda, alpha):
        super(EGAT_EGNN, self).__init__()
        self.appnp_nlayers = egat_nlayers
        self.egnn_nlayers = egnn_nlayers
        self.appnp_baseModules = nn.ModuleList()
        self.egnn_baseModules = nn.ModuleList()
        for _ in range(egat_nlayers):
            self.appnp_baseModules.append(APPNPBaseModule(alpha=APPNP_ALPHA))
        for _ in range(egnn_nlayers):
            self.egnn_baseModules.append(EGNNBaseModule(in_size=nhidden2, hidden_size=nhidden2, out_size=nhidden2, edge_feat_size=2))
        self.egnn_scale_projs = nn.ModuleList([nn.Linear(nhidden2, nhidden1) for _ in range(egnn_nlayers)])
       
        self.egnn_scale_pool_logits = nn.Parameter(torch.zeros(egnn_nlayers))
        self.alpha_head = nn.Linear(nhidden1, 1)

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden1))
        self.fcs.append(nn.Linear(nfeat, nhidden2))
        self.fcs.append(nn.Linear(egat_nlayers * nhidden1 + nhidden1, nhidden1))
        self.fcs.append(nn.Linear(nhidden1, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        fusion_dim = egat_nlayers * nhidden1 + nhidden1
        self.sa = MUSEAttention(d_model=fusion_dim, d_k=fusion_dim, d_v=fusion_dim, h=1)


    def forward(self, x, graph=None, efeats=None, adj_matrix=None, pos=None, return_hidden=False):
        appnp_layer_inner = F.dropout(self.act_fn(self.fcs[0](x)), self.dropout, training=self.training)
        egnn_layer_inner = F.dropout(self.act_fn(self.fcs[1](x)), self.dropout, training=self.training)
        appnp_h0 = appnp_layer_inner

       
        egnn_scale_states = []
        for i,egnn_baseMod in enumerate(self.egnn_baseModules):
            egnn_layer_inner, pos = egnn_baseMod(
                input=egnn_layer_inner,
                coord_feat=pos,
                graph=graph,
                efeats=efeats,
            )
            egnn_layer_inner = F.dropout(egnn_layer_inner, self.dropout, training=self.training)
            egnn_scale_states.append(self.act_fn(self.egnn_scale_projs[i](egnn_layer_inner)))

        egnn_scale_tensor = torch.stack(egnn_scale_states, dim=0)  
        egnn_scale_weights = torch.softmax(self.egnn_scale_pool_logits, dim=0).view(-1, 1, 1)  
        egnn_multiscale = (egnn_scale_tensor * egnn_scale_weights).sum(dim=0)

      
        appnp_out = []
        for i, appnp_baseMod in enumerate(self.appnp_baseModules):
            if self.appnp_nlayers == 1:
                scale_idx = self.egnn_nlayers - 1
            else:
                scale_idx = int(round(i * (self.egnn_nlayers - 1) / (self.appnp_nlayers - 1)))
           
            geo_context = 0.5 * (egnn_scale_states[scale_idx] + egnn_multiscale)
            alpha_adaptive = torch.sigmoid(self.alpha_head(geo_context))
            appnp_layer_inner = appnp_baseMod(
                input=appnp_layer_inner,
                h0=appnp_h0,
                adj_matrix=adj_matrix,
                adaptive_alpha=alpha_adaptive,
            )
            appnp_layer_inner = F.dropout(appnp_layer_inner, self.dropout, training=self.training)
            appnp_out.append(appnp_layer_inner)

        appnp_layer_inner = torch.cat(appnp_out, dim=1)
        layer_inner = torch.cat([appnp_layer_inner, egnn_multiscale], dim=1)
        layer_inner = torch.unsqueeze(layer_inner,dim=0)
        layer_inner = self.sa(layer_inner, layer_inner, layer_inner)
        layer_inner = torch.squeeze(layer_inner, dim=0)
        hidden = self.fcs[-2](layer_inner)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.act_fn(hidden)
        logits = self.fcs[-1](hidden)
        if return_hidden:
            return logits, hidden
        return logits


class SGAPPIS(nn.Module):
    def __init__(
        self,
        nfeat,
        nhidden1,
        nhidden2,
        nclass,
        dropout,
        lamda,
        alpha,
        egat_layers=None,
        egnn_layers=None,
    ):
        super(SGAPPIS, self).__init__()
      
        self.deep_agat = EGAT_EGNN(
            egat_nlayers=APPNP_LAYER if egat_layers is None else int(egat_layers),
            egnn_nlayers=EGNN_LAYER if egnn_layers is None else int(egnn_layers),
            nfeat=nfeat,
            nhidden1=nhidden1,
            nhidden2=nhidden2,
            nclass=nclass,
            dropout=dropout,
            lamda=lamda,
            alpha=alpha,
        )

        self.criterion = focal_loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=1e-6,
        )

    def forward(self, x, graph, adj_matrix, pos):
        x = x.float()
        x = x.view([x.shape[0] * x.shape[1], x.shape[2]])
        pos = pos.float()
        output = self.deep_agat(
            x=x,
            graph=graph,
            efeats=graph.edata['ex'],
            adj_matrix=adj_matrix,
            pos=pos,
            return_hidden=False,
        )
        return output
