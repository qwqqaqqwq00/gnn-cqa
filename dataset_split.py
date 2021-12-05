import json
import os
import time
from tqdm import tqdm
import numpy as np
import random
import sys
import warnings

warnings.filterwarnings('ignore')
sys.setrecursionlimit(4000)

def dataset_stats(dataset_questions_json_path, save_repo):
    with open(dataset_questions_json_path) as js:
        file = json.load(js)
    total = len(file)
    no_dup = 0
    no_rel = 0
    no_dup_and_rel = 0
    at_least_one = 0
    max_dups = 0
    max_rel = 0
    bad = 0

    for qid, content in file.items():
        if content['dups'] == []:
            no_dup += 1
        if content['related'] == []:
            no_rel += 1
        if content['related'] == [] and content['dups'] == []:
            no_dup_and_rel += 1
        if content['related'] or content['dups']:
            at_least_one += 1
        if len(content['dups'])>max_dups:
            max_dups = len(content['dups'])
        if len(content['related'])>max_rel:
            max_rel = len(content['related'])

        if content['dups']:
            for i in content['dups']:
                if qid not in file[i]['dups']:
                    if qid not in file[i]['related']:
                        bad += 1
        if content['related']:
            for i in content['related']:
                if qid not in file[i]['related']:
                    if qid not in file[i]['dups']:
                        bad += 1

    num_dups = np.zeros(max_dups+1, dtype = np.int32)
    num_rel = np.zeros(max_rel+1, dtype = np.int32)

    for qid, content in file.items():
        num_dups[len(content['dups'])] += 1
        num_rel[len(content['related'])] += 1

    with open(os.path.join(save_repo,'Dataset_INFO.txt'), 'w') as t:
        t.write("total: "+str(total)+"\n")
        t.write("no dup: "+str(no_dup)+"\n")
        t.write("no rel: "+str(no_rel)+"\n")
        t.write("no dup and no rel: "+str(no_dup_and_rel)+"\n")
        t.write("have dup or have rel: "+str(at_least_one)+"\n")
        t.write("if have dup, max number of dups: "+str(max_dups)+", and its distribution: "+"\n")
        num_dups=[str(line)+"," for line in num_dups.tolist()]
        t.writelines(num_dups)
        t.write('\n')
        t.write("if have rel, max number of rels: "+str(max_rel)+", and its distribution: "+"\n")
        num_rel = [str(line)+"," for line in num_rel.tolist()]
        t.writelines(num_rel)
        t.write('\n')
        t.write("single arrows: "+str(bad)+"\n")

def findByRow(mat, row):
    return np.where((mat == row).all(1))[0]

def edge_save(edge, split, flag, save_repo):
    split_edge = np.zeros((1,3),dtype = np.int32)
    temp = np.zeros((1,3),dtype = np.int32)

    for i in split:
        idx = np.argwhere(np.logical_or(edge[:,0]==i,edge[:,1]==i)).flatten()
        for j in idx:
            if not findByRow(split_edge, edge[j,:]):
                split_edge = np.concatenate((split_edge, np.expand_dims(edge[j,:],axis=0)),axis=0)

    split_edge = np.delete(split_edge,0,axis=0)
    np.savetxt(os.path.join(save_repo,'split_'+str(flag)+'_edge.csv'), split_edge, delimiter=",",fmt="%d,%d,%d")


def node_dfs(edge, id, split):
    if id not in split:
        split.append(id)
        idx = np.argwhere(edge[:,0]==id).flatten()
        if idx is not []:
            for i in idx:
                node_dfs(edge, edge[i][1], split)
        idx_2 = np.argwhere(edge[:,1]==id).flatten()
        if idx_2 is not []:
            for j in idx_2:
                node_dfs(edge, edge[j][0], split)

def node_write(split, path):
    with open(path,'w') as t:
        for i in split:
            t.write(str(i)+'\n')

def dataset_spilt(dataset_questions_json_path, save_repo):
    with open(dataset_questions_json_path) as js:
        file = json.load(js)
        
    all_questions = []
    relavant_questions = []
    all_q_num = len(file)
    edge = np.zeros((1,3),dtype = np.int32)
    temp = np.zeros((1,3),dtype = np.int32)
    
    for i, content in file.items():
        all_questions.append(int(i))
        for j in content['dups']:
            assert int(i) != int(j) , "Error: duplicate question is itself."
            greater = 0
            smaller = 0
            if int(i)>int(j):
                greater = int(i)
                smaller = int(j)
            else:
                greater = int(j)
                smaller = int(i)
            idx = np.argwhere(np.logical_and(edge[:,0]==smaller,edge[:,1]==greater)).squeeze()
            if not idx:
                temp[0][0] = smaller
                temp[0][1] = greater
                temp[0][2] = 2
                edge = np.concatenate((edge,temp),axis=0)
            
        for j in content['related']:
            assert int(i) != int(j) , "Error: related question is itself."
            greater = 0
            smaller = 0
            if int(i)>int(j):
                greater = int(i)
                smaller = int(j)
            else:
                greater = int(i)
                smaller = int(j)
            idx = np.argwhere(np.logical_and(edge[:,0]==smaller,edge[:,1]==greater)).squeeze()
            if not idx:
                temp[0][0] = smaller
                temp[0][1] = greater
                temp[0][2] = 1
                edge = np.concatenate((edge,temp),axis=0)
    
    edge = np.delete(edge,0,axis=0)
    np.savetxt(os.path.join(save_repo, 'all_edge.csv'), edge, delimiter=",",fmt="%d,%d,%d")

    for i in range(edge.shape[0]):
        if edge[i][0] not in relavant_questions:
            relavant_questions.append(edge[i][0])
        if edge[i][1] not in relavant_questions:
            relavant_questions.append(edge[i][1])
    random.shuffle(relavant_questions)

    split_1 = []
    split_2 = []
    split_3 = []
    split_4 = []

    k = 0
    while(len(split_1)<len(relavant_questions)//2):
        node_dfs(edge,relavant_questions[k],split_1)
        k += 1

    split_3 = list(set(relavant_questions) - set(split_1))
   
    k = 0
    while(len(split_2)<len(split_1)//2):
        node_dfs(edge,split_1[k],split_2)
        k += 1
    
    split_1 = list(set(split_1) - set(split_2))

    k = 0
    while(len(split_4)<len(split_3)//2):
        node_dfs(edge,split_3[k],split_4)
        k += 1
    
    split_3 = list(set(split_3) - set(split_4))    

    with open(os.path.join(save_repo,'Splits_INFO.txt'),'w') as f:
        f.write("Splits INFO:\n")
        f.write("split_1_rel: "+str(len(split_1))+"\n")
        f.write("split_2_rel: "+str(len(split_2))+"\n")
        f.write("split_3_rel: "+str(len(split_3))+"\n")
        f.write("split_4_rel: "+str(len(split_4))+"\n")

    edge_save(edge, split_1, 1, save_repo)
    edge_save(edge, split_2, 2, save_repo)
    edge_save(edge, split_3, 3, save_repo)
    edge_save(edge, split_4, 4, save_repo)

    non_relavant_questions = list(set(all_questions) - set(relavant_questions))
    random.shuffle(non_relavant_questions)

    split_1.extend(non_relavant_questions[:len(non_relavant_questions)//4])
    split_2.extend(non_relavant_questions[len(non_relavant_questions)//4:len(non_relavant_questions)//2])
    split_3.extend(non_relavant_questions[len(non_relavant_questions)//2:len(non_relavant_questions)*3//4])
    split_4.extend(non_relavant_questions[len(non_relavant_questions)*3//4:])

    random.shuffle(split_1)
    random.shuffle(split_2)
    random.shuffle(split_3)
    random.shuffle(split_4)

    with open(os.path.join(save_repo,'Splits_INFO.txt'),'a') as f:
        f.write("split_1_all: "+str(len(split_1))+"\n")
        f.write("split_2_all: "+str(len(split_2))+"\n")
        f.write("split_3_all: "+str(len(split_3))+"\n")
        f.write("split_4_all: "+str(len(split_4))+"\n")


    node_write(split_1, os.path.join(save_repo,"split_1.txt"))
    node_write(split_2, os.path.join(save_repo,"split_2.txt"))
    node_write(split_3, os.path.join(save_repo,"split_3.txt"))
    node_write(split_4, os.path.join(save_repo,"split_4.txt"))


if __name__ == '__main__':
    path = r"./cqadupstack"
    save_path = r"./data"
    themes = [theme for theme in os.listdir(path) if os.path.isdir(os.path.join(path, theme)) ]
    with tqdm(total=len(themes),ncols=100) as pbar:
        pbar.set_description("Spilt Process")
        for theme_name in themes:
            json_file = os.path.join(path, theme_name, theme_name+"_questions.json")
            save_repo = os.path.join(save_path,theme_name)
            if not os.path.exists(save_repo):
                os.makedirs(save_repo)
            dataset_stats(json_file, save_repo)
            dataset_spilt(json_file, save_repo)
            pbar.update(1)