import json
import os
import bs4
# import nltk
import pickle
import warnings
import re
import time
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer,BertModel

from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import torch
import argparse
import numpy as np

class Embeding():
    def __init__(self):
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.to(args.device_ids[0])
        self.encoder.eval()
        self.vocab = {'[SEP]':0, '[CLS]':1}
        self.vocab_sz = 2
        self.long_sent = 0
    
    def bert_embed(self, st):
        with torch.no_grad():
            ids = []
            for word in st:
                pos = self.vocab.get(word)
                if pos == None:
                    self.vocab[word] = self.vocab_sz
                    ids.append(self.vocab_sz)
                    self.vocab_sz += 1
                else:
                    ids.append(pos)
            if len(ids) > 512:
                long_sent = "long" + str(self.long_sent)
                self.vocab[long_sent] = self.vocab_sz
                ids = [0, self.vocab_sz, 0]
                self.long_sent += 1
                self.vocab_sz += 1

            Input_ids = torch.LongTensor(ids).unsqueeze(0)

            Input_ids = Input_ids.to(args.device_ids[0])
            sen_out, _ = self.encoder(Input_ids)
            ret = sen_out[-1].to('cpu').numpy()
    
            # all_encoder_layers, _ = self.encoder(st)
        # all_encoder_layers return the output of * layers transformer
        # pooled_output return the sentence embedding
        
        return ret
    
    def tokenize(self, body, auto=True):
        body = re.sub('[.,!?]', ' ', body)
        sens = self.snowball(body)
        if auto == True:
            vecs = []
            vecs.append(self.bert_embed(sens))
            if len(vecs) == 0:
                vecs = [np.zeros((0, 768))]
            vecs = np.concatenate(vecs, 0)
            return vecs
        else:
            return sens

    def snowball(self, sen):
        token = word_tokenize(sen)
        stm = SnowballStemmer('english')
        res = [stm.stem(word) for word in token]
        res = [word for word in res if word not in stopwords.words('english')]
        return  res
