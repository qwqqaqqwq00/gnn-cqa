import json
import os

dataset_questions_json_path = r'./cqadupstack/gis/gis_questions.json'

with open(dataset_questions_json_path) as js:
    file = json.load(js)

vocab = {}
no_tag = 0

for qid, content in file.items():
    if content['tags']:
        for tag in content['tags']:
            if tag not in vocab:
                vocab[tag] = 1
            else:
                vocab[tag] += 1
    else:
        no_tag += 1


with open('temp1.txt','w') as t:
    t.write('There are '+str(no_tag)+' questions which do not have any tag.\n\n')
    for k,v in vocab.items():
        t.write(str(k)+':\t'+str(v)+'\n')
