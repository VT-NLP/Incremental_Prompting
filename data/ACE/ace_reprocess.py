import json

ace_train_path = "ace_origin/ACE.train.jsonl"
ace_dev_path = "ace_origin/ACE.dev.jsonl"
ace_test_path = "ace_origin/ACE.test.jsonl"
processed_train = "./ACE.train.jsonl"
processed_dev = "./ACE.dev.jsonl"
processed_test = "./ACE.test.jsonl"

train_ids = set()
dev_ids = set()
test_ids = set()
training_list = {}
dev_list = {}
test_list = {}

if __name__ == '__main__':
    # with open(ace_train_path, "rt") as atp:
    #     for tins in atp:
    #         train_ins = json.loads(tins)
    #         train_ids.add(train_ins["sentence_id"])

    with open(ace_test_path, "rt") as file:
        for ins in file:
            instance = json.loads(ins)
            if len(instance["piece_ids"]) > 342:
                continue
            _id = instance["sentence_id"]
            if _id not in dev_list.keys():
                dev_list[_id] = {"piece_ids": instance["piece_ids"], "label": [], "span": [], "sentence_id": _id, "mention_id": []}
                dev_list[_id]["label"].append(instance["label"])
                dev_list[_id]["span"].append(instance["span"])
                dev_list[_id]["mention_id"].append(instance["mention_id"])
            else:
                dev_list[_id]["label"].append(instance["label"])
                dev_list[_id]["span"].append(instance["span"])
                dev_list[_id]["mention_id"].append(instance["mention_id"])
        # print(len(dev_list))

    with open(processed_test, "wt") as fp:
        for _id, dev in dev_list.items():
            fp.write(json.dumps(dev)+"\n")
