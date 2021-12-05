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


warnings.filterwarnings('ignore')

# userID_index
_answers_index   = [
    "body",
    "userid",
    "comments",
    "score",
    "parentid", 
    "creationdate"]  # so do comments
_questions_index = [
    "body",
    "viewcount",
    "dups",
    "title",
    "tags",
    "userid",
    "related",
    "score",
    "answers",
    "acceptedanswer",
    "creationdate",
    "favoritecount",
    "comments"]
_user_index      = [
    "views",
    "rep",
    "lastaccessdate",
    "answers",
    "age",
    "questions",
    "upvotes",
    "downvotes",
    "badges",
    "date_joined"]
_words           = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
'//', "''", '>>', '>>>', '|', '&&', '/*', '*/', '\\', '--', '__',
'an', 'the', 'do', 'this', 'it', 'he', 'her', 'me', 'you', 'to', 'in', 'at', 'from', 'that',
'these', 'those', 'its', 'his', 'mine', 'your', 'yours'
]

class Embeding():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.to(args.device_ids[0])
        self.encoder.eval()
        # self.vocab = {'[SEP]':0, '[CLS]':1}
        # self.vocab_sz = 2
        # self.long_sent = 0
    
    def filter(self, text):
        text = re.sub("[\n\r\t\d]", " ", text)
        text = re.sub("<pre><code>.*</code></pre>", " [UNK] ", text)
        text = re.sub("\$.*\$|\{.*?\}|\[.*?\]|\(.*?\)|\<.*?\>", " ", text)
        return text
    
    def bert_embed(self, st):
        with torch.no_grad():
            ids = []
            # for word in st:
                # if word in _words:
                    # continue
                # pos = self.vocab.get(word)
                # if pos == None:
                #     if self.vocab_sz >=30500:
                #         self.vocab_sz = 2
                #     self.vocab[word] = self.vocab_sz
                #     ids.append(self.vocab_sz)
                #     self.vocab_sz += 1
                # else:
                #     ids.append(pos)
            token = self.tokenizer.tokenize(st)
            for word in _words:
                if word in token:
                    token.remove(word)
            ids = self.tokenizer.convert_tokens_to_ids(token)
            if len(ids) > 512:
                # long_sent = "long" + str(self.long_sent)
                # self.vocab[long_sent] = self.vocab_sz
                # ids = [0, self.vocab_sz, 0]
                # self.long_sent += 1
                # self.vocab_sz += 1
                ids = ids[:512]
            elif len(ids) == 0:
                return np.zeros((1, 0, 768), dtype=np.float64)

            Input_ids = torch.LongTensor(ids).unsqueeze(0)

            Input_ids = Input_ids.to(args.device_ids[0])
            sen_out, _ = self.encoder(Input_ids)
            ret = sen_out[-1].to('cpu').numpy()
    
            # all_encoder_layers, _ = self.encoder(st)
        # all_encoder_layers return the output of * layers transformer
        # pooled_output return the sentence embedding
        
        return ret
    
    def tokenize(self, body, auto=True):
        body = re.sub("['~`@#$%^&*(_[\\])+=\{\}\/\n;:\t<->1234567890]", ' ', body)
        # sens = self.snowball(body)
        sens = sent_tokenize(body)
        if auto == True:
            vecs = []
            for sen in sens:
                vecs.append(self.bert_embed(sen))
            if len(vecs) == 0:
                vecs = [np.zeros((1, 0, 768), dtype=np.float64)]
            vecs = np.concatenate(vecs, 1)
            return vecs
        else:
            return body

    def snowball(self, sen):
        token = word_tokenize(sen)
        stm = SnowballStemmer('english')
        res = [stm.stem(word) for word in token]
        res = [word for word in res if word not in stopwords.words('english')]
        return  res


def walk_to(paths):
    json_file = {}
    paths_index = {}
    for path in os.walk(paths):
        for ph in path[2]:
            title = ph.split('.')[0]
            path_value = path[0] + "/" + ph
            paths_index[title] = path_value

    for title, path in paths_index.items():
        with open(path) as f:
            file = json.load(f)
            json_file[title] = file
    return json_file


def question_content_process(qid, content, bt):
    body = bt.filter(content['body'])
    soup = bs4.BeautifulSoup(body)
    body = soup.get_text()
    body = bt.tokenize(body)
    title = content['title']
    title = bt.tokenize(title)
    dups = content['dups']
    tags = content['tags']
    related = content['related']

    return {"id": qid, "body": body, "title": title, "tags": tags, "related": related, "dups": dups}

# def user_content_process(content):
#     # answers = bs4.BeautifulSoup(content['answers'])
#     # answers = answers.get_text()
#     # answers = nltk.tokenize.word_tokenize(answers)
#     # questions = bs4.BeautifulSoup(content['questions'])
#     # questions = questions.get_text()
#     # questions = nltk.tokenize.word_tokenize(questions)
#     answers = content['answers']
#     questions = content['questions']
#     return {"ans": answers, "ques": questions}


def preprocess(path_theme):
    json_files = walk_to(path_theme)
    
    answer = {}
    comment = {}
    question = {}
    user = {}

    file_index = os.listdir(path_theme)
    k = 0
    if (k for k in range(len(file_index)) if file_index[k].find("questions")) != 0:
        file_index[k], file_index[0] = file_index[0], file_index[k]
    theme_name = os.path.basename(path_theme)

    bt = Embeding()

    with open("Log.txt", "a") as log:
        log.write(theme_name+":\n")

    for fileindex in file_index:
        fileindex = os.path.splitext(fileindex)[0]
        with open("Log.txt", "a") as log:
            log.write(fileindex+":"+str(len(json_files[fileindex]))+"\n")
        with tqdm(total=len(json_files[fileindex])) as pbar:
            pbar.set_description(fileindex)
            # if fileindex == theme_name+"_answers":
            #     i = 0
            #     for userid, content in json_files[fileindex].items():
            #         answer[i] = answers_content_process(i, content, bt)
            #         i += 1
            #         pbar.update(1)
            # elif fileindex == theme_name+"_comments":
            #     for userid, content in json_files[fileindex].items():
            #         # comment[userid] = comment_content_process(content, bt)
            #         # pbar.update(1)
            #         pass
            if fileindex == theme_name+"_questions":
                question = []
                for userid, content in json_files[fileindex].items():
                    question.append(question_content_process(userid, content, bt))
                    pbar.update(1)
            # elif fileindex == theme_name+"_users":
            #     for userid, content in json_files[fileindex].items():
            #         user[userid] = user_content_process(content)
            #         pbar.update(1)
            else:
                print(fileindex+" could not match any of {answers, comments, questions, users}.")
                pbar.update(len(json_files[fileindex]))
        
        
    return question


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", default="3", type=lambda x: list(map(int, x.split(','))),
                help="Names of the devices comma separated.")
    args = parser.parse_args()
    path = r"./cqadupstack"
    for theme_name in [theme for theme in os.listdir(path) if os.path.isdir(os.path.join(path, theme)) ]:
    # if 1:
        # theme_name = "android"
        data = preprocess(os.path.join(os.path.join(path, theme_name)))
        with open(theme_name + ".pkl", "wb") as pps:
            pickle.dump(data, pps)
