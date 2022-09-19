import json
from tqdm import tqdm

raw_file_path = "unprocess/train.jsonl"
train_mentions_file = "unprocess/maven.train_mentions.jsonl"
dev_mentions_file = "unprocess/maven.dev_mentions.jsonl"

train_mentions = set()
dev_mentions = set()
training_list = []
dev_list = []

if __name__ == '__main__':
    # read mentions
    with open(train_mentions_file, "rt") as tmf:
        for tmen in tmf:
            train_mention = json.loads(tmen)
            train_mentions.add(train_mention["sentence_id"].split('_')[0])
    print("train doc size")
    print(len(train_mentions))
    with open(dev_mentions_file, "rt") as dmf:
        for dmen in dmf:
            dev_mention = json.loads(dmen)
            dev_mentions.add(dev_mention["sentence_id"].split('_')[0])
    print("dev doc size")
    print(len(dev_mentions))
    with open(raw_file_path, "rt") as fp:
        for document_line in tqdm(fp):
            document = json.loads(document_line)
            if document["id"] in train_mentions:
                training_list.append(document)
            elif document["id"] in dev_mentions:
                dev_list.append(document)
    with open("unprocess/dev_split.jsonl", "wt") as fp:
        for dev in dev_list:
            fp.write(json.dumps(dev)+"\n")
    with open("unprocess/train_split.jsonl", "wt") as fp:
        for train in training_list:
            fp.write(json.dumps(train)+"\n")
