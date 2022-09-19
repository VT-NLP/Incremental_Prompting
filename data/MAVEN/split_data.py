import json
import os
import random
from tqdm import tqdm


random.seed(2147483647)

num_doc = 413
num_training_doc = 2913
label_start_offset = 1
num_cls = 168

sample_doc_ids = random.sample(range(1, num_training_doc), num_doc)
sample_doc_ids.append(8)
sample_doc_ids = set(sample_doc_ids)
print(len(sample_doc_ids))
contained_type = []
total_document_list = []
training_list = []
dev_list = []
file_path = "unprocess/train.jsonl"


def _file():
    with open(file_path, "rt") as fp:
        for document_line in tqdm(fp):
            document = json.loads(document_line)
            total_document_list.append(document)

def _process(doc_list):
    for doc_id in sample_doc_ids:
        _document(doc_list[doc_id])


def _document(document):
    events = document["events"]
    for event in events:
        label = event['type_id']
        contained_type.append(label)


def _split():
    for i, doc in enumerate(total_document_list):
        if i in sample_doc_ids:
            dev_list.append(doc)
        else:
            training_list.append(doc)
    with open("dev_split.jsonl", "wt") as fp:
        for dev in dev_list:
            fp.write(json.dumps(dev)+"\n")
    with open("train_split.jsonl", "wt") as fp:
        for train in training_list:
            fp.write(json.dumps(train)+"\n")


def check_missed():
    # missed id: 88
    _file()
    _process(total_document_list)
    label_dict_list = []
    cover_types = set(contained_type)
    for i in range(1, num_cls+1):
        if i in cover_types:
            label_dict_list.append({i: 0})
        else:
            label_dict_list.append({i: 1})
    print(cover_types)
    print(len(cover_types))
    print(label_dict_list)

if __name__ == '__main__':
    # check_missed()
    _file()
    _split()




