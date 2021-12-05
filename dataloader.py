import pickle
import os
import numpy as np
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
from torch_geometric.typing import Adj
from encoder import Encoder
import time
from tqdm import tqdm
from encoder import Model

theme_name = [
    # "android",
    "english",
    # "gaming",
    # "gis",
    # "mathematica",
    # "physics",
    # "programmers",
    # "stats",
    # "tex",
    # "unix",
    # "webmasters",
    # "wordpress"
]

device = 3
epochs = 20
win = 32
split_files =True
# early_dropout = 1000000

def split_file(i):
    print("batch of data "+get_theme_name+"'s "+str(i)+" batch.")
    # v1, v2, relation
    edge_tuple = np.loadtxt(data_path + get_theme_name + "/split_"+str(i)+"_edge.csv", delimiter=",", dtype=np.int64)
    # with open(path + "data/" + get_theme_name + "/split_"+str(i)+".csv", "r") as split:

    vertex = np.loadtxt(path + "data/" + get_theme_name + "/split_" + str(i)+".txt", dtype=np.int64)
    # for lines in open(path + "data/" + get_theme_name + "/split_" + str(i)+".txt", "r"):
    td = t.Tensor([])
    x = 0
    # require_data = []
    edge_dup = t.Tensor([])
    edge_rel = t.Tensor([])
    edge2idx = t.Tensor([])
    idx2edge = {}
    print("try to load data...")
    pbar = tqdm(total=len(data))
    pbar.set_description("data iter")
    for j in range(len(data)):
        # if j > early_dropout:
            # break
        tmp_d = vertex[vertex == int(data[j]['id'])]
        pbar.update(1)
        # pbar = enumerate(tqdm(j, desc="block bar"), 0)
        if len(tmp_d) > 0:
            x += 1
            # slice_data = data[j]
            # require_data.append(slice_data)
            recv = t.from_numpy(data[j]["body"])
            # idx_tmp = t.Tensor([data[j]['id']])
            # edge2idx = t.cat([edge2idx, idx_tmp], 0)
            idx2edge[data[j]['id']] = j
            if len(data[j]['related']) > 0:
                for it in data[j]['related']:
                    edge_tmp = t.Tensor([[int(data[j]['id'])], [int(it)]])
                    edge_rel = t.cat([edge_rel, edge_tmp],1)
            if len(data[j]['dups']) > 0:
                for it in data[j]['dups']:
                    edge_tmp = t.Tensor([[int(data[j]['id'])], [int(it)]])
                    edge_dup = t.cat([edge_dup, edge_tmp],1)
            if len(recv[0]) > 512:
                recv = recv[:, :512, :]
                
            recv_s = t.zeros((1, 512, 768))
            recv_len = recv.size(1)
            recv_s[:, :recv_len, :] += recv
            td = t.cat([td, recv_s], dim=0)
    print("catch the graph, is preprocessing.")
    print("dup edge process...")
    pbar = tqdm(total=edge_dup.size(1))
    for k in range(edge_dup.size(1)):
        if idx2edge.get(edge_dup[0][k].item()) is not None:
            edge_dup[0][k] = idx2edge[edge_dup[0][k].item()]
        if idx2edge.get(edge_rel[1][k].item()) is not None:
            edge_dup[1][k] = idx2edge[edge_dup[1][k].item()]
        pbar.update(1)
    pbar = tqdm(total=edge_rel.size(1))
    for k in range(edge_rel.size(1)):
        if idx2edge.get(edge_rel[0][k].item()) is not None:
            edge_rel[0][k] = idx2edge[edge_rel[0][k].item()]
        if idx2edge.get(edge_rel[1][k].item()) is not None:
            edge_rel[1][k] = idx2edge[edge_rel[1][k].item()]
        pbar.update(1)
    print("saving processed file.")
    t.save({
        "data":td,
        "dups":edge_dup,
        "rels":edge_rel
    }, str(i)+"tensor.pth")
    with open("idx2edge"+str(i)+".pkl", "wb") as eg:
        pickle.dump(idx2edge, eg)


if __name__ == "__main__":
    path = r"./"
    data_path = r"./data/"
    model = Model(win - 1, 1, 3, 768, 384, 511, 8).cuda(device)
    optimizer = t.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=1e-3
        )
    t.autograd.set_detect_anomaly(True)
    model.train()
    for get_theme_name in theme_name:
        print("loading data from "+get_theme_name)
        with open(path + get_theme_name + ".pkl", "rb") as pps:
            data = pickle.load(pps)
        
        for i in range(1, 5):
            if split_files:
                split_file(i)
            else:
                struct_tensor = {
                    "data":1,
                    "dups":2,
                    "rels":3
                }
                load_tensors = t.load(str(i)+"tensor.pth")
                td = load_tensors['data']
                edge_dup = load_tensors['dups']
                edge_rel = load_tensors['rels']
                with open("idx2edge"+str(i)+".pkl", "rb") as eg:
                    idx2edge = pickle.load(eg)
            
            if i <= 4:
                for epoch in range(epochs):
                    pbar = tqdm(total=td.size(0))
                    pbar.set_description("training process...")
                    for k in range(int(td.size(0)/win)):
                        window = td[k * win:(k+1)*win]
                        optimizer.zero_grad()
                        for l in range(window.size(0)):
                            dup_idx = (edge_dup == l).nonzero()
                            dup = edge_dup[:, dup_idx[:, 0]]
                        for l in range(window.size(0)):
                            rel_idx = (edge_rel == l).nonzero()
                            rel = edge_rel[:, rel_idx[:, 0]]
                        score = [dup.cuda(device), rel.cuda(device)]
                        window_s = window.permute(0, 2, 1).cuda(device) # (b, d, l)
                        loss = model(window_s[0:1], window_s[1:], score)
                        loss.requires_grad_(True)
                        loss.backward()
                        optimizer.step()
                        pbar.update(1)
                        window_s.cpu()
                        dup.cpu()
                        rel.cpu()
                    t.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        },
                        str(i)+str(epoch)+"model.pth"
                    )
                    print("current loss:"+str(loss.item()))
            
            elif i > 4:
                t.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    str(i)+"model.pth"
                )
                model.eval()
                item = 0
                l = 0
                for k in range(int(data.size(0) / win)):
                    l+=1
                    window = data[k:k+win]
                    score = [edge_dup[k:k+win].cuda(device), edge_rel[k:k+win].cuda(device)]
                    window_s = window.permute(0, 2, 1).cuda(device)
                    loss = model(window_s[0:1], window_s[1:], score, 'eval')
                    item += loss.item()
                #eval
                print("eval:"+str(item/l))
