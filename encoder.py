from math import pi, exp
import math
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import activation
import torch_geometric as tm
from torch_geometric.data import Data
from torch.autograd import Variable
import numpy as np
from torch.nn import init
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.message_passing import MessagePassing
# from torch_geometric.nn.conv.transformer_conv import TransformerConv
# from conv_transformer import TransformerConv
# from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from gat_v2_conv import GATv2Conv
from torch_geometric.nn.pool.sag_pool import SAGPooling
from torch_geometric.typing import Adj

device = 3

class Encoder(nn.Module):
    def __init__(self, input_len, ques_len, cls_dim, hid_dim=768, g_hid_dim=384, sens_len=512, heads=8, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        """
        Parameters:
            quary:R{n,d,l}
            ques :R{m,d,l}
        """
        self.quary_len = input_len
        self.ques_len = ques_len
        self.batch_size = self.quary_len + self.ques_len
        self.cls_dim = cls_dim
        self.hid_dim = hid_dim
        self.sens_len = sens_len
        self.heads = heads
        self.g_hid_dim = g_hid_dim
        self.sen_len_trans = [sens_len - 1, int((sens_len - 1) / 4), int((sens_len - 1) / 32), self.cls_dim + 1]
        self.init_sentence()
        self.init_graph()

        self.cauchy_param = t.Tensor([2.]).cuda(device)
        nn.Parameter(self.cauchy_param, requires_grad=True)


    def init_sentence(self):
        self.encoderlayer1 = nn.TransformerEncoderLayer(self.hid_dim, self.heads, activation='gelu', batch_first=True).cuda(device) #self.hid_dim=512->sen_len batch_first = True
        self.encoder1 = nn.TransformerEncoder(self.encoderlayer1, 6).cuda(device)
        self.lin1 = nn.Sequential(
            nn.Linear(self.sens_len + 1, self.sens_len + 1), 
            nn.Dropout(0.2)
        ).cuda(device)
        self.posm = [
            MessageMaxSelector(self.batch_size, self.hid_dim, self.heads).cuda(device),
            MessageMaxSelector(self.batch_size, self.hid_dim, self.heads).cuda(device),
            MessageMaxSelector(self.batch_size, self.hid_dim, self.heads).cuda(device), 
            MessageMaxSelector(self.batch_size, self.hid_dim, self.heads).cuda(device),
        ]
        self.lin = [
            nn.Sequential(
                nn.Linear(self.sen_len_trans[i], self.sen_len_trans[i]),
                nn.Dropout(0.2)
            ).cuda(device) for i in range(4)
        ]
        self.norm1 = nn.BatchNorm2d(self.cls_dim + 1).cuda(device)
        self.norm2 = nn.BatchNorm1d(self.batch_size).cuda(device)
        self.regularizer = Regularizer(1, 5e-2, 1e9)
        # self.rnn = nn.Sequential([
        #     nn.RNN(self.sens_len, self.sens_len / 2, self.hid_dim),
        #     nn.Linear(self.cls_dim, self.cls_dim),
        #     nn.ReLU()
        # ]).cuda(device)
        

    def init_graph(self):
        # self.para_init()
        self.conv1 = GCNConv(int(self.g_hid_dim), int(self.g_hid_dim)).cuda(device)
        # self.conv2 = TransformerConv(int(self.hid_dim / 2), int(self.hid_dim / 2)).cuda(device)
        # self.conv3 = TransformerConv(int(self.hid_dim / 2), int(self.hid_dim / 2)).cuda(device)
        self.conv4 = GCNConv(int(self.g_hid_dim), int(self.g_hid_dim)).cuda(device)

    def adjacency(self, idx: t.Tensor=t.zeros((0,))):
        node2idx = self.node0
        if idx != None:
            node2idx = t.cat([self.node0, idx], dim=0)
        idx2node = {}
        for node, idx in node2idx:
            idx2node[idx] = node
        ad_mat = Variable(t.zeros((node2idx.size(0), node2idx.size(0)))).cuda(device)

        for i in self.edge_index.permute(1, 0):
            if idx2node.get(i[0]) is not None and idx2node.get(i[1]) is not None:
                a, b = idx2node[i[0]], idx2node[i[1]]
                ad_mat[a][b] = 1
                ad_mat[b][a] = 1
            
        if idx != None:
            for i in range(len(idx)):
                ad_mat = ad_mat[:self.node0.size(0)][self.node0.size(0):]
            
        return ad_mat

    def cauchy_selector(self, x):
        base = t.sqrt(t.sum(t.pow(x, 2), dim=2).unsqueeze(2))
        out = t.div(t.bmm(x, x.permute(0, 2, 1)), 
                    t.bmm(base, base.permute(0, 2, 1))
        )
        out -= t.eye(self.batch_size).unsqueeze(0).repeat((self.cls_dim, 1, 1)).cuda(device)
        # out = t.pow(out, t.exp(self.cauchy_param))
        pos = 0.5 - 0.5 * t.cos(pi * out)
        out += pos
        return out
    
    # def para_init(self):
        # init.uniform_(self.gamma)

    
    def normal_edge(self, adj_, sigma=0.5):
        data = []
        adj = self.norm2(t.sum(adj_, dim=-1))
        i = 0
        for adj_i in adj:
            adj_edge = (adj_i > sigma).type(t.float)
            edge_index = adj_edge.nonzero().t()
            row, col = edge_index
            edge_weight = adj_[i, row, col, :]
            # x = t.diagonal(adj_edge @ adj_edge).unsqueeze(1).type(t.float)
            x = t.diagonal(adj_[i]).permute(1, 0)
            data.append(Data(x=x, edge_index=edge_index.contiguous(), edge_weight=edge_weight).cuda(device))
            i += 1
        return data

    def posm_box(self, q, i):
        q = self.posm[i](q, last_mask=True)  # in:(b,d,l) out:(b,l,d)
        q, mask = q[:, :, :-1], q[:, :, -1:]
        mask += t.sum(q[:, :, self.sen_len_trans[i] - 1:], -1).unsqueeze(-1)
        q = q[:, :, :self.sen_len_trans[i] - 1]
        q = t.cat([q, mask], -1)
        q = self.lin[i](q)
        return q

    def coo_edge(self, x):
        adj = []
        assert self.hid_dim % self.g_hid_dim == 0
        for i in range(self.g_hid_dim):
            adj.append(self.cauchy_selector(x[:, :, i:i + 2]).unsqueeze(-1))
        # for i in range(adj.size(0)):
            # adj[i, t.eye(adj.size(1), dtype=t.bool)] = 0
        # adj = self.norm2(adj.unsqueeze(0)).squeeze(0)
        adj = t.cat(adj, -1)
        return self.normal_edge(adj)

    def _build_sentence(self, input:t.Tensor, ques:t.Tensor):
        q = t.cat([input, ques], dim=0)
        q, mask = q[:, :, :-1], q[:, :, -1:]
        q = q.permute(0, 2, 1) # n, l, d
        # print(q.size())
        q = self.encoder1(q).permute(0, 2, 1)
        q = t.cat([q, mask], dim=-1)
        q = self.lin1(q)
        for i in range(4):
            q = self.posm_box(q, i)

        q = q.permute(2, 0, 1)
        q = self.norm1(q.unsqueeze(0)).squeeze(0)
        q = q.permute(1, 2, 0)
        q, label = q[:, :, :-1], q[:, :, -1:] # (b, d, l)
        # q = q.permute(0, 2, 1)
        data = self.coo_edge(q.permute(2, 0, 1))
        return  data, label

    def _build_prob_graph_sample(self, data):
        for i in range(len(data)):
            data[i].x += self.conv1(data[i].x, data[i].edge_index)
            data[i].x += self.conv4(data[i].x, data[i].edge_index)
        return data

    def forward(self, input: t.Tensor, ques: t.Tensor):
        data, label = self._build_sentence(input, ques)
        data = self._build_prob_graph_sample(data)
        # data = self.graph_models(data)
        return data, label

class MessageMaxSelector(nn.Module):
    def __init__(self, batch, in_features, trans_layers=8):
        """
        x:(b,d,l)->batch,dim,length
        in_feature:dim
        trans_layers:(dim/*)!=0
        """
        super(MessageMaxSelector, self).__init__()
        self.batch = batch
        layers = [in_features, in_features]
        self.lin1 = nn.Sequential(
            nn.Conv1d(in_features, layers[0], 1),
            *[nn.Conv1d(layers[i], layers[i + 1], 1) for i in range(len(layers) - 1)],
            nn.Conv1d(layers[-1], in_features, 1)
        ).cuda(device)
        self.lin2 = nn.Sequential(
            nn.Conv1d(in_features, layers[0], 1),
            *[nn.Conv1d(layers[i], layers[i + 1], 1) for i in range(len(layers) - 1)],
            nn.Conv1d(layers[-1], in_features, 1)
        ).cuda(device)
        self.lin3 = nn.Sequential(
            nn.Conv1d(in_features, layers[0], 1),
            *[nn.Conv1d(layers[i], layers[i + 1], 1) for i in range(len(layers) - 1)],
            nn.Conv1d(layers[-1], in_features, 1)
        ).cuda(device)
        self.lin4 = nn.Sequential(
            nn.Conv1d(in_features, layers[0], 1),
            *[nn.Conv1d(layers[i], layers[i + 1], 1) for i in range(len(layers) - 1)],
            nn.Conv1d(layers[-1], in_features, 1)
        ).cuda(device)
        # self.trans = nn.TransformerEncoderLayer(in_features, trans_layers, activation='gelu', batch_first=True) #(b, l, h)->(b, l, t) , batch_first=True
        # self.encoder = nn.TransformerEncoder(self.trans, trans_layers)
        # self.alpha = t.Tensor([1])
        # self.beta = t.Tensor([1])
        # nn.Parameter(self.alpha, requires_grad=True)
        # nn.Parameter(self.beta, requires_grad=True)

    def idx_select(self, x, idx):
        x_view = x.view(x.size(0)*x.size(1),-1).unsqueeze(1)
        y = t.zeros_like(x_view).cuda(device)
        for i in range(idx.size(0)):
            for j in range(idx.size(1)):
                y[i][j][idx[i][j]] = x[i][j][idx[i][j]]
        return y

    def cauchy_selector(self, sx, cx, cauchy_param=1e-2):
        out = t.div(t.pow(t.bmm(sx, cx), 2),
                    t.bmm(t.sum(t.pow(sx, 2), dim=2).unsqueeze(2), t.sum(t.pow(cx, 2), dim=1).unsqueeze(1)))
        out = t.pow(out, exp(cauchy_param))
        out -= t.eye(sx.size(1)).unsqueeze(0).repeat((self.batch, 1, 1)).cuda(device)
        out = 0.5 - 0.5 * t.cos(pi * out)
        # out = t.clip(out, 0, 1)
        return out

    def complex_selector(self, sx, cx, px):
        # (b, d, l)
        sx, cx = t.complex(sx, t.zeros_like(sx)), t.complex(cx, t.zeros_like(cx))
        out = t.div(t.bmm(sx, cx),
                    t.bmm(t.sqrt(t.sum(t.pow(sx, 2), dim=2)).unsqueeze(2), t.sqrt(t.sum(t.pow(cx, 2), dim=1)).unsqueeze(1)))
        out -= t.eye(sx.size(1)).unsqueeze(0).repeat((sx.size(0), 1, 1)) * (1 + 1.0j)
        out = t.exp(t.arccos(pi * out / 2) * 2.0j)
        out_real = t.bmm(t.softmax(out.real, dim=2), px)
        out_imag = t.bmm(t.softmax(out.imag, dim=2), px)
        out = t.cat([out_real, out_imag], dim=1)
        return out

    def forward(self, x:t.Tensor, last_mask=False):
        # batch, dim = x.size(0), x.size(1)
        if last_mask is True:
            x, mask = x[:, :, :-1], x[:, :, -1:]
        dsx = self.lin1(x)
        dcx = self.lin2(x)
        lsx = self.lin3(x)
        lcx = self.lin4(x)
        # score, score_idx = t.topk(score, k=1, dim=1) #(b*d, 1, lp)
        dscore = t.softmax(F.relu(self.cauchy_selector(dsx, dcx.permute(0, 2, 1))), dim=1)  # (b, ls, lp)
        lscore = t.softmax(F.relu(self.cauchy_selector(lsx.permute(0, 2, 1), lcx)), dim=2)
        # score_x = self.idx_select(x, score_idx) # (b*d, d, ls)
        # score = score.reshape(batch, dim, -1)
        # score = x * score # (b, d, ls)
        y = (t.bmm(t.bmm(dscore, x), lscore) + x) / 2
        if last_mask is True:
            # score = t.cat([score, mask], -1)
            y = t.cat([y, mask], -1)  #(b,d,l)
        return y

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        if entity_embedding is None:
            return None
        return t.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class ProjectionKL(nn.Module):
    def __init__(self, vertex_dim, hid_dim):
        super(ProjectionKL, self).__init__()
        self.vertex_dim = vertex_dim
        self.hid_dim = hid_dim
        self.layer0 = nn.Linear(vertex_dim, hid_dim).cuda(device)
        # self.layer1 = nn.Linear(vertex_dim, hid_dim).cuda(device)
        self.layer1 = nn.Linear(hid_dim, vertex_dim).cuda(device)

    def _kl_calc(self, alpha, beta):
        out = []
        if alpha.size(0) < 1:
            return alpha
        for i in range(alpha.size(0)):
            out.append(alpha[i] + t.sum(F.kl_div(alpha[i].unsqueeze(0).sigmoid().log(), beta.sigmoid(), reduction='none'), 0).unsqueeze(0))
        out = t.cat(out, 0)
        return out

    def forward(self, alpha, beta):
        a_size = alpha.size(0)
        if a_size == 0:
            out = self.layer0(beta)
            return self.layer1(out)
        else:
            out = t.cat([alpha, beta], 0)
            out = self.layer0(out)
            alpha = self._kl_calc(out[a_size:, :], out[:a_size, :])
            out = t.cat([alpha, out[:a_size, :]], 0)
            out = self.layer1(out)
            return out[a_size:, :], out[:a_size, :]

class Intersection(nn.Module):
    def __init__(self, vertex_dim, edge_dim, hid_dim, batch_size):
        super(Intersection, self).__init__()
        self.vertex_dim = vertex_dim
        self.batch_size = batch_size
        self.hid_dim = hid_dim
        self.edge_dim = edge_dim
        # self.aggregator = MessagePassing("add", "source_to_target", -2, 1)
        self.layer0 = nn.Linear(vertex_dim, hid_dim).cuda(device)
        self.w_0 = nn.Linear(edge_dim, hid_dim).cuda(device)
        self.layer1 = nn.Linear(hid_dim, vertex_dim).cuda(device)
        self.w_1 = nn.Linear(hid_dim, edge_dim).cuda(device)
        self.norm = nn.BatchNorm1d(hid_dim).cuda(device)

    def _cauchy_selector(self, x):
        base = t.sqrt(t.sum(t.pow(x, 2), dim=1).unsqueeze(1))
        out = t.div(t.matmul(x, x.permute(1, 0)), 
                    t.matmul(base, base.permute(1, 0))
        )
        out -= t.eye(out.size(-1)).cuda(device)
        return out
    
    # def _adjacency(self, idx: t.Tensor=t.zeros((0,))):
    #     col, row = idx
    #     ad_mat = t.zeros((self.batch_size, self.batch_size)).cuda(device)
    #     ad_mat[row, col] = 1
    #     ad_mat[col, row] = 1
    #     return ad_mat

    def forward(self, x, edge_index, edge_weight, damping=0.45, path=3):
        out = self.layer0(x)
        w = self.w_0(edge_weight)
        
        for i in range(self.batch_size):
            idx = (edge_index[0] == i).nonzero()
            if idx.size(0) == 0:
                continue
            else:
                select_w = w[idx, :]
                out[i:i+1, :] += damping * t.sum(select_w, 0)
                w[idx, :] += damping * t.sum(out[i, :], 0).unsqueeze(0)
        out_ = t.softmax(self._cauchy_selector(out), 1)
        out = t.matmul(out_, out)
        if w.size(0) > 1:
            w = self.norm(w)
        return self.layer1(out), self.w_1(w)

class KGComplete(MessagePassing):
    def __init__(self, edge_dim, cls_dim, batch_size, vertex_dim=1, hid_dim=256, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(KGComplete, self).__init__(**kwargs)
        # self.num_vertex = num_vertex
        self.edge_dim = edge_dim
        self.vertex_dim = vertex_dim
        self.hid_dim = hid_dim
        self.batch_size = batch_size
        self.cls_dim = cls_dim
        self._graph_init()

    def _graph_init(self):
        self.regularizer = Regularizer(1, 5e-2, 1e9)
        # self.intersection = BetaIntersection(self.edge_dim).cuda(device)
        # self.projection = BetaProjection(self.vertex_dim, self.edge_dim, self.hid_dim, self.regularizer, self.cls_dim).cuda(device)
        self.projection = ProjectionKL(self.vertex_dim, self.hid_dim)
        self.intersection = Intersection(self.vertex_dim, self.edge_dim, self.hid_dim, self.batch_size)
        self.conv = GATv2Conv(self.vertex_dim, self.vertex_dim, dropout=0.3).cuda(device)
        self.norm = nn.BatchNorm1d(self.edge_dim)
        self.normx = nn.BatchNorm1d(self.vertex_dim)
        self.lin1 = nn.Linear(self.edge_dim, self.edge_dim)
        self.lin2 = nn.Linear(self.vertex_dim, self.vertex_dim)
        self.sagp = SAGPooling(self.vertex_dim, min_score=1e-3, GNN=GCNConv).cuda(device)
        self.relu = nn.ReLU()

    def _nagetive(self, input):
        return t.sigmoid(1./input)

    def _adjacency(self, idx: t.Tensor=t.zeros((0,))):
        col, row = idx
        ad_mat = t.zeros((self.batch_size, self.batch_size)).cuda(device)
        ad_mat[row, col] = 1
        ad_mat[col, row] = 1
        return ad_mat

    def _beta_quary_embed(self, data):
        rel = {'dup':0, 'rel':1, 'none':2}
        alpha = []
        beta = []
        dup_rel = []
        for i in range(self.cls_dim):
            if i == 0:
                tmpa, tmpb = self.intersection(data[i].x, data[i].edge_index, data[i].edge_weight)
                data[i].x+=tmpa
                data[i].edge_weight += tmpb
                data[i].x = self.relu(data[i].x)
                adj = self._adjacency(data[i].edge_index)
                idx = (t.sum(adj, 0) < 1).nonzero()
                nidx = (t.sum(adj, 1) >= 1).nonzero()
                if idx.size(0) > 0:
                    negative = rel['none']
                    al, be = self.projection(data[i].x[idx[:, 0]], data[i].x[nidx[:, 0]])
                    data[negative].x += t.cat([al,be],0)
                    tmpa,tmpb= self.intersection(self._nagetive(data[negative].x), data[i].edge_index, self._nagetive(data[i].edge_weight))
                    data[negative].x+=tmpa
                    data[negative].edge_weight += t.mean(tmpb, 0)
                # positive (duplicate)
                data[i].x = self.conv(x=data[i].x, edge_index=data[i].edge_index)
                x, edge_index, _, batch, perm, score = self.sagp(data[i].x, data[i].edge_index)
                data[i].x, data[i].edge_index = x, edge_index
                data[i].x = self.relu(data[i].x)
                row, col = data[i].edge_index
                select_embed = t.cat([row, col], dim=0)
                dup_rel.append(select_embed)
                embeding = data[i].x[select_embed, :]
                a_embeding, b_embeding = t.chunk(embeding, 2, 1)
                alpha.append(a_embeding)
                beta.append(b_embeding)
        # data[0].x, data[0].edge_index, _, batch, perm, score = self.sagp(data[0].x, data[0].edge_index, data[0].edge_weight, None)
            elif i == 1:
                # offset
                data[i].edge_weight = self.lin1(data[i].edge_weight)
                if data[i].edge_weight.size(0) > 1:
                    data[i].edge_weight = self.norm(data[i].edge_weight)
                data[i].x = self.lin2(data[i].x)
                data[i].x = self.normx(data[i].x)
                tmpa, tmpb = self.intersection(data[i].x, data[i].edge_index, data[i].edge_weight)
                data[i].x += tmpa
                data[i].edge_weight += tmpb
                data[i].x = self.relu(data[i].x)
                adj = self._adjacency(data[i].edge_index)
                idx = (t.sum(adj, 0) < 1).nonzero()
                nidx = (t.sum(adj, 1) >= 1).nonzero()
                if idx.size(0) > 0:
                    negative = rel['none']
                    al, be = self.projection(data[i].x[idx[:, 0]], data[i].x[nidx[:, 0]])
                    data[i].x += t.cat([al,be],0)
                    tmpa, tmpb = self.intersection(self._nagetive(data[negative].x), data[i].edge_index, self._nagetive(data[i].edge_weight))
                    data[negative].x += tmpa
                    data[negative].edge_weight += t.mean(tmpb, 0)
                # positive (relate)
                data[i].x = self.conv(x=data[i].x, edge_index=data[i].edge_index)
                x, edge_index, _, batch, perm, score = self.sagp(data[i].x, data[i].edge_index)
                data[i].x, data[i].edge_index = x, edge_index
                data[i].x = self.relu(data[i].x)
                row, col = data[i].edge_index
                select_embed = t.cat([row, col], dim=0)
                dup_rel.append(select_embed)
                embeding = data[i].x[select_embed, :]
                a_embeding, b_embeding = t.chunk(embeding, 2, 1)
                alpha.append(a_embeding)
                beta.append(b_embeding)
            elif i == 2:
                # none
                data[i].x = self._nagetive(data[i].x)
                tmpa, tmpb = self.intersection(data[i].x, data[i].edge_index, data[i].edge_weight)
                data[i].x += tmpa
                data[i].edge_weight += tmpb
                data[i].x = self.relu(data[i].x)
                data[i].x = self.conv(data[i].x, data[i].edge_index)
                x, edge_index, _, batch, perm, score = self.sagp(data[i].x, data[i].edge_index)
                data[i].x, data[i].edge_index = x, edge_index
                data[i].x = self.relu(data[i].x)
                idx = []
                dup_rel = t.cat(dup_rel, 0)
                for it in range(data[i].x.size(0)):
                    if it not in dup_rel:
                        idx.append(it)
                a_embeding, b_embeding = t.chunk(data[i].x[idx, :], 2, 1)
                alpha.append(a_embeding)
                beta.append(b_embeding)   
        return alpha, beta, data

    def forward(self, out):
        return self._beta_quary_embed(out)

class CalcLoss(nn.Module):
    def __init__(self, batch):
        super(CalcLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean').cuda(device)
        self.gamma = t.Tensor([24.]).cuda(device)
        nn.Parameter(self.gamma, requires_grad=True)
        self.batch_size = batch
    
    def _adjacency(self, idx: t.Tensor=t.zeros((0,))):
        if idx.size(1) < 1:
            return t.zeros((self.batch_size, self.batch_size)).cuda(device)
        col, row = idx
        ad_mat = t.zeros((self.batch_size, self.batch_size)).cuda(device)
        ad_mat[row, col] = 1
        ad_mat[col, row] = 1
        return ad_mat

    def loss(self, alpha, beta, data, score, mode='train'):
        dup_rel = t.cat([
            self._adjacency(data[0].edge_index).unsqueeze(-1),
            self._adjacency(data[1].edge_index).unsqueeze(-1),
            self._adjacency(data[2].edge_index).unsqueeze(-1)
        ], -1)
        dup_rel = (dup_rel.softmax(-1)[:, :, 0:2] > 0.5).float()
        a_score = t.cat([
            self._adjacency(score[0]).unsqueeze(-1),
            self._adjacency(score[1]).unsqueeze(-1)
        ], -1)
        map_loss = self.l1(dup_rel, a_score)
        if mode == 'eval':
            return map_loss
        # logit = t.max(self.cal_logit_beta(alpha, beta))
        return map_loss #+ logit

    def cal_logit_beta(self, alpha, beta):
        logit = 0
        if alpha[0].size(0) > 0 and alpha[1].size(0) > 0:
            a, b = t.chunk(alpha[0], 2, 0)
            c, d = t.chunk(beta[0], 2, 0)
            dist_0 = t.distributions.beta.Beta(t.clamp(a, 1e-2, 1e9), t.clamp(c, 1e-2, 1e9))
            dist_1 = t.distributions.beta.Beta(t.clamp(b, 1e-2, 1e9), t.clamp(d, 1e-2, 1e9))
    #     # if alpha[2].size(0) > 0:
    #         # dist_2 = t.distributions.beta.Beta(t.sum(t.clamp(alpha[2], 1e-2, 1e9), dim=0), t.sum(t.clamp(beta[2], 1e-2, 1e9), dim=0))
    #     # else:
    #         # dist_2 = t.distributions.beta.Beta(t.sum(t.clamp(t.cat([alpha[0], alpha[1]], 0), 1e-2, 1e9), 0), t.sum(t.clamp(t.cat([beta[0], beta[1]], 0), 1e-2, 1e9), 0))
            
            logit = t.norm(t.distributions.kl.kl_divergence(dist_0, dist_1), p=1, dim=-1) + self.gamma
    #     # logit += self.gamma - t.norm(t.distributions.kl.kl_divergence(dist_1, dist_2), p=1, dim=-1)
    #     # logit += self.gamma - t.norm(t.distributions.kl.kl_divergence(dist_0, dist_2), p=1, dim=-1)
        return logit
    # def cal_logit_cauthy(self, alpha, beta):


    def forward(self, alpha, beta, data, score, mode='train'):
        return self.loss(alpha, beta, data, score, mode)

class Model(nn.Module):
    def __init__(self, input_len, ques_len, cls_dim, hid_dim=768, g_hid_dim=384, sens_len=512, heads=8):
        super(Model, self).__init__()
        self.en = Encoder(input_len, ques_len, cls_dim, hid_dim, g_hid_dim, sens_len, heads).cuda(device)
        self.kg = KGComplete(g_hid_dim, cls_dim, input_len+ques_len, g_hid_dim, g_hid_dim).cuda(device)
        self.logit = CalcLoss(input_len+ques_len).cuda(device)

    def forward(self, quary, ques, score, mode='train'):
        out, label = self.en(quary, ques)
        a, b, data = self.kg(out)
        loss = self.logit(a, b, data, score, mode)
        return loss

    def autograph(self, quary, ques):
        out,label = self.en(quary, ques)
        return self.kg(out)

import time
if __name__ == "__main__":
    model = Model(10, 1, 3, 768, 384, 512, 8)
    quary = t.randn((1, 768, 513)).cuda(device) #(b, d, l)
    ques = t.randn((10, 768, 513)).cuda(device)
    score = [t.randint(0, 10, (2, 24)), t.randint(0, 10, (2, 25))]
    st = time.time()
    t.autograd.set_detect_anomaly(True)
    for i in range(1400):
        loss = model(quary, ques, score)
        loss.requires_grad_(True)
        print(loss)
        loss.backward()
    print(time.time()-st, "s")
    # print(label.size())
