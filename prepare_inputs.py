from typing import Any, List, Tuple, Union, Callable
from transformers import BertTokenizerFast
import json
import os
from tqdm import tqdm
import torch
import numpy as np
import transformers

class Instance(object):
    '''
    - piece_ids: L
    - label: 1
    - span: 2
    - sentence_id: str
    - mention_id: str
    '''
    def __init__(self, piece_ids: List[int], label: List[int], span: List[Tuple[int, int]], sentence_id: str,
                 mention_id: List[str]) -> None:
        self.piece_ids = piece_ids
        self.label = label
        self.span = span
        self.sentence_id = sentence_id
        self.mention_id = mention_id

    def todict(self,):
        return {
            "piece_ids": self.piece_ids,
            "label": self.label,
            "span": self.span,
            "sentence_id": self.sentence_id,
            "mention_id": self.mention_id
        }


class MAVENPreprocess(object):

    def __init__(self, root, tokenizer, label_start_offset=1, max_length=512, expand_context=False, split_valid=True):
        super().__init__()
        train_file = os.path.join(root, "train.jsonl")
        dev_file = os.path.join(root, "dev.jsonl")
        test_file = os.path.join(root, "test.jsonl")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expand_context = expand_context
        self.label_start_offset = label_start_offset
        self.label_ids = {}
        self.collected = set()
        self.model = None
        train_instances = self._file(train_file)
        dev_instances = self._file(dev_file)
        test_instances = self._file(test_file)

        with open("data/MAVEN/MAVEN.train.jsonl", "wt") as fp:
            for instance in train_instances:
                fp.write(json.dumps(instance.todict())+"\n")
        with open("data/MAVEN/MAVEN.dev.jsonl", "wt") as fp:
            for instance in dev_instances:
                fp.write(json.dumps(instance.todict())+"\n")
        with open("data/MAVEN/MAVEN.test.jsonl", "wt") as fp:
            for instance in test_instances:
                fp.write(json.dumps(instance.todict())+"\n")

    def _file(self, file_path):
        instances = []
        with open(file_path, "rt") as fp:
            for document_line in tqdm(fp):
                document = json.loads(document_line)
                instances.extend(self._document(document))
        return instances

    # modified for dealing with multiple span
    def _document(self, document):
        document_id = document["id"]
        title = document['title']
        sentences = document["content"]
        events = document["events"]
        none_events = document["negative_triggers"]
        instances = []
        labels = [[] for _ in range(len(sentences))]
        spans = [[] for _ in range(len(sentences))]
        mention_ids = [[] for _ in range(len(sentences))]
        sentence_ids = ["" for _ in range(len(sentences))]
        piece_list = [[] for _ in range(len(sentences))]

        for event in events:
            label = self.label_start_offset + event['type_id']
            if event['type'] not in self.label_ids:
                self.label_ids[event['type']] = label
            for mention in event['mention']:
                sentence = sentences[mention['sent_id']]
                sentence_id = f"{document_id}_{mention['sent_id']}"
                sent_id = mention['sent_id']
                span = mention["offset"]
                mention_id = mention["id"]
                piece_ids, span = self._transform_single(
                    token_ids=sentence["tokens"],
                    spans=[span[0],span[0], span[1]-1, span[1]-1],
                    tokenizer=self.tokenizer,
                    is_tokenized=True)

                if len(piece_list[sent_id]) == 0 and len(piece_ids) <= 342:
                    piece_list[sent_id].extend(piece_ids)
                    sentence_ids[sent_id] = sentence_id
                span = (span[0], span[3])
                spans[sent_id].append(span)
                labels[sent_id].append(label)
                mention_ids[sent_id].append(mention_id)

        for mention in none_events:
            sentence = sentences[mention['sent_id']]
            sentence_id = f"{document_id}_{mention['sent_id']}"
            span = mention["offset"]
            mention_id = mention["id"]
            piece_ids, span = self._transform_single(
                token_ids=sentence["tokens"],
                spans=[span[0],span[0], span[1]-1, span[1]-1],
                tokenizer=self.tokenizer,
                is_tokenized=True)
            sent_id = mention['sent_id']

            if len(piece_list[sent_id]) == 0:
                piece_list[sent_id].extend(piece_ids)
                sentence_ids[sent_id] = sentence_id
            span = (span[0], span[3])

            spans[sent_id].append(span)
            labels[sent_id].append(0)
            mention_ids[sent_id].append(mention_id)

        for i in range(len(sentences)):
            if len(piece_list[i]) >= 512 or len(piece_list[i]) <= 2:    # ignore overlength or empty instance
                continue
            instance = Instance(
                piece_ids=piece_list[i],
                label=labels[i],
                span=spans[i],
                sentence_id=sentence_ids[i],
                mention_id=mention_ids[i])
            instances.append(instance)

        return instances

    def _context(self, sentences:List[List[str]]) -> List[Tuple[List[int], int, int]]:
        raise NotImplementedError


    @classmethod
    def _transform_single(cls, token_ids: Union[List[List[str]], List[str], str], spans: Union[List[int], Tuple[int]], tokenizer: BertTokenizerFast, is_tokenized: bool=False) -> Tuple[List[int], List[int]]:
        def _token_span(cls, offsets, s, e):
            ts = []
            i = 0
            while offsets[i][0] <= s:
                i += 1
            ts.append(i - 1)
            i -= 1
            while offsets[i][1] <= e:
                i += 1
            ts.append(i)
            return tuple(ts)
        sent_id = hs = he = ts = te = 0
        _token_ids = _spans = []
        if len(spans) == 4:
            hs, he, ts, te = spans
        else:
            sent_id, hs, he, ts, te = spans
        if isinstance(token_ids, str):
            if is_tokenized:
                raise TypeError("Cannot process single string when 'is_tokenized = True'.")
            else:
                tokens = tokenizer(token_ids, return_offsets_mapping=True)
                _token_ids = tokens["input_ids"]
                offsets = tokens["offset_mapping"][1:-1]
                h = _token_span(offsets, hs, he)
                t = _token_span(offsets, ts, te)
                _spans = [h[0] + 1, h[1] + 1, t[0] + 1, t[1] + 1]
        elif isinstance(token_ids, List):
            if is_tokenized:
                tokens = tokenizer(token_ids, is_split_into_words=True, return_offsets_mapping=True)
                if isinstance(token_ids[0], str):
                    _token_ids = tokens["input_ids"]
                    offsets = tokens["offset_mapping"]
                    token2piece = []
                    piece_idx = 1
                    for x, y in offsets[1:-1]:
                        if x == 0:
                            if len(token2piece) > 0:
                                token2piece[-1].append(piece_idx-1)
                            token2piece.append([piece_idx])
                        piece_idx += 1
                    if len(token2piece[-1]) == 1:
                        token2piece[-1].append(piece_idx-1)
                    _spans = [token2piece[hs][0], token2piece[he][1], token2piece[ts][0], token2piece[te][1]]
                else:
                    token2piece = []
                    piece_idx = 1
                    for x, y in tokens["offset_mapping"][sent_id][1:-1]:
                        if x == 0:
                            if len(token2piece) > 0:
                                token2piece[-1].append(piece_idx-1)
                            token2piece.append([piece_idx])
                        piece_idx += 1
                    if len(token2piece[-1]) == 1:
                        token2piece[-1].append(piece_idx-1)
                    _spans = [token2piece[hs][0], token2piece[he][1], token2piece[ts][0], token2piece[te][1]]
                    _token_ids = []
                    for i, t in enumerate(tokens["input_ids"]):
                        if i == sent_id:
                            _spans = [_t - 1 + len(_token_ids) for _t in _spans]
                        if i > 0:
                            _token_ids.extend(t[1:])
                        else:
                            _token_ids.extend(t)
            else:
                tokens = tokenizer(token_ids, return_offsets_mapping=True)
                if isinstance(token_ids[0], str):
                    offsets = tokens["offset_mapping"][sent_id][1:-1]
                    h = _token_span(offsets, hs, he)
                    t = _token_span(offsets, ts, te)
                    _spans = [h[0], h[1], t[0], t[1]]
                    _token_ids = []
                    for i, t in enumerate(tokens["input_ids"]):
                        if i == sent_id:
                            _spans = [_t + len(_token_ids) for _t in _spans]
                        if i > 0:
                            _token_ids.extend(t[1:])
                        else:
                            _token_ids.extend(t)
                else:
                    raise TypeError("Cannot process list of lists of sentences (list of paragraphs).")

        return _token_ids, _spans


def main():
    MAVEN_PATH = "./data/MAVEN/" # path for original maven dataset
    bt = BertTokenizerFast.from_pretrained("bert-large-cased")
    m1 = MAVENPreprocess(MAVEN_PATH, tokenizer=bt)


if __name__ == "__main__":
    main()
