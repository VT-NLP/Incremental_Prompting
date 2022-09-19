from transformers import BertTokenizerFast
import json
import re

file_path = "../data/ACE/label2btid.json"
out_file_path = "../data/ACE/id2tokens.json"
# streams = [[134, 99, 105, 91, 52, 19, 135, 122, 127, 102, 108, 104, 110, 38, 118, 121, 160, 26, 34, 33, 158, 130, 165, 86, 149, 146, 164, 4, 131, 75, 90, 10, 32, 0], [46, 36, 17, 13, 12, 20, 25, 49, 169, 148, 87, 56, 95, 81, 58, 57, 161, 64, 163, 82, 43, 125, 35, 96, 156, 100, 66, 30, 50, 120, 0], [147, 48, 84, 39, 53, 23, 109, 22, 77, 62, 93, 143, 89, 8, 71, 85, 63, 97, 65, 103, 113, 112, 123, 152, 88, 98, 72, 42, 60, 133, 128, 116, 15, 111, 136, 162, 129, 132, 6, 0], [157, 18, 114, 92, 67, 153, 80, 115, 78, 154, 27, 5, 69, 117, 44, 83, 155, 40, 45, 139, 137, 14, 138, 119, 74, 150, 59, 54, 28, 9, 140, 37, 107, 76, 47, 0], [151, 16, 3, 70, 106, 141, 68, 31, 79, 101, 166, 167, 94, 21, 51, 7, 73, 145, 2, 124, 159, 61, 41, 142, 126, 144, 55, 168, 29, 11, 24, 0]]
streams = [[188, 199, 184, 194, 189, 190, 178, 186, 197, 0], [191, 187, 195, 171, 174, 179, 0], [201, 183, 193, 172, 175, 0], [200, 180, 173, 198, 177, 0], [196, 181, 182, 202, 185, 176, 170, 192, 0]]
if __name__ == '__main__':
    bt = BertTokenizerFast.from_pretrained("bert-large-cased")
    with open(file_path, "rt") as fp:
        label2id = json.load(fp)
        print(label2id)
    # print("-----num of event types-----", len(label2id))
    type_tids = []
    id2tokens = {0: [100]}
    for type_name in label2id.keys():
        tn = type_name.lower()
        tns = re.split(':|-', tn)
        type_tid = bt.convert_tokens_to_ids(tns)
        if isinstance(type_tid, int):
            type_tid = [type_tid]
        id2tokens[int(label2id[type_name])] = type_tid
        type_tids.append(type_tid)
    recover_names = []
    for type_tid in type_tids:
        tokens = bt.convert_ids_to_tokens(type_tid)
        recover_names.append(tokens)

    task_event_num = [len(stream) for stream in streams]
    print(task_event_num)
    with open(out_file_path, "w") as out_file:
        json.dump(id2tokens, out_file)
    print(type_tids)
    print(id2tokens)
    for recover_name in recover_names:
        for ren in recover_name:
            print(ren, end=" ")
        print()
