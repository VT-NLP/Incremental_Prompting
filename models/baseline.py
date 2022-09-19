import json
import numpy
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch import autograd
from transformers import BertModel
import math
from typing import Any, Dict, Tuple, List, Union, Set
import warnings
from collections import OrderedDict
from torch.nn.modules.linear import Linear
from torchmeta.modules import MetaLinear, MetaSequential, MetaModule, MetaBilinear
from tqdm import tqdm
from utils.options import parse_arguments
import random

opts = parse_arguments()
BERT_VOCAB_SIZE = 28996
BERT_MAXLEN = 512

PROMPT_SIZE = opts.prompt_size
device = torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))

random.seed(opts.seed)


class KDR(MetaModule):
    def __init__(self, input_dim: int, hidden_dim: int, max_slots: int, init_slots: int, label_mapping: dict,
                 device: Union[torch.device, None] = None, **kwargs) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-large-cased')
        for name, param in list(self.bert.named_parameters()):
            param.requires_grad = False
        self.half_input_dim = int(input_dim / 2)
        self.input_map = MetaSequential(OrderedDict({
            "linear_0": MetaLinear(input_dim, hidden_dim),
            "relu_0": nn.ReLU(),
            "dropout_0": nn.Dropout(0.2),
            "linear_1": MetaLinear(hidden_dim, hidden_dim),
            "relu_1": nn.ReLU()
        }))

        self.classes = MetaLinear(hidden_dim, max_slots, bias=False)
        _mask = torch.zeros(1, max_slots, dtype=torch.float, device=device)
        _mask[:, init_slots:] = float("-inf")
        self.register_buffer(name="_mask", tensor=_mask)
        self.crit = nn.CrossEntropyLoss()
        self.device = device
        self.to(device=device)
        self.nslots = init_slots
        self.max_slots = max_slots
        self.maml = True
        self.outputs = {}
        self.history = None
        self.exemplar_input = None
        self.exemplar_features = None
        self.exemplar_attm = None
        self.exemplar_labels = None
        self.exemplar_span = None
        self.exemplar_prompt = None
        self.exemplar_size = None
        self.random_exemplar_inx = None
        self.dev_exemplar_features = None
        self.dev_exemplar_labels = None

        self.iter_cnt = 0
        self.period = opts.period
        self.e_weight = opts.eloss_w



    @property
    def mask(self, ):
        self._mask[:, :self.nslots] = 0
        self._mask[:, self.nslots:] = float("-inf")
        return self._mask

    def idx_mask(self, idx: Union[torch.LongTensor, int, List[int], None] = None,
                 max_idx: Union[torch.LongTensor, int, None] = None):
        assert (idx is not None) or (max_idx is not None)
        assert (idx is None) or (max_idx is None)
        mask = torch.zeros_like(self._mask) + float("-inf")
        if idx is not None:
            mask[:, idx] = 0
        if max_idx is not None:
            if isinstance(max_idx, torch.LongTensor):
                max_idx = max_idx.item()
            mask[:, :max_idx] = 0
        return mask

    @property
    def features(self):
        return self.classes.weight[:self.nslots]

    @staticmethod
    def avg_span(encoding, span_mask):
        s_mask = span_mask.unsqueeze(1)
        span_len = (span_mask != 0).sum(dim=1).unsqueeze(1)
        s_sum = torch.bmm(s_mask.float(), encoding).squeeze(1)
        s_avg = s_sum.float() / span_len.float()
        return s_avg

    def forward(self, batch, nslots: int = -1, exemplar: bool = False, exemplar_distill: bool = False,
                feature_distill: bool = False, mul_distill=False, distill: bool = False, return_loss: bool = True,
                return_feature: bool = False, tau: float = 1.0, log_outputs: bool = True, params=None, task_id: int = 0,
                store: bool = False):

        self.iter_cnt += 1
        input_ids, attention_masks, labels, spans, prompt_masks = batch.token_ids, batch.attention_masks, batch.labels, \
                                                                  batch.spans, batch.prompt_masks
        span_bsz = spans.size(0)
        if store:
            span_input = input_ids.unsqueeze(0).repeat(span_bsz, 1, 1)
            span_attm = attention_masks.repeat(span_bsz, 1, 1)
            self.outputs["input"] = span_input.detach().cpu()
            self.outputs["attm"] = span_attm.detach().cpu()
            self.outputs["prom"] = prompt_masks.detach().cpu()

        outputs = self.bert(input_ids.unsqueeze(0), attention_mask=attention_masks.unsqueeze(0))
        enc_outputs = outputs[0]
        bsz, seq_len, hidden_dim = enc_outputs.shape
        span_bsz = spans.size(0)
        rep_enc_outputs = enc_outputs.repeat(span_bsz, 1, 1)

        span_idx = spans.unsqueeze(-1)
        span_idx = span_idx.expand(-1, -1, hidden_dim)
        span_repr = torch.gather(rep_enc_outputs, 1, span_idx)
        # input feature: (span_bsz, hidden_dim*2)
        features = span_repr.view(span_bsz, hidden_dim * 2)

        inputs = self.input_map(features, params=self.get_subdict(params, "input_map"))
        scores = self.classes(inputs, params=self.get_subdict(params, "classes"))

        if torch.any(torch.isnan(scores)):
            print(scores[0])
            input('a')
        if nslots == -1:
            scores += self.mask
            if torch.any(torch.isnan(scores)):
                print(scores[0])
                input()
            nslots = self.nslots
        else:
            scores += self.idx_mask(max_idx=nslots)
        scores[:, 0] = 0

        if scores.size(0) != labels.size(0):
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        else:
            labels = labels
        if log_outputs:
            pred = torch.argmax(scores, dim=1)
            acc = torch.mean((pred == labels).float())
            self.outputs["accuracy"] = acc.item()
            self.outputs["prediction"] = pred.detach().cpu()
            self.outputs["label"] = labels.detach().cpu()
            self.outputs["spans"] = spans.detach().cpu()
            self.outputs["in_features"] = features.detach().cpu()
            self.outputs["encoded_features"] = inputs.detach().cpu()

        if return_loss:
            labels.masked_fill_(labels >= nslots, 0)
            valid = labels < nslots
            nvalid = torch.sum(valid.float())
            if nvalid == 0:
                loss = 0
            else:
                loss = self.crit(scores[valid], labels[valid])
                if torch.isnan(loss):
                    print(labels, nslots, scores[:, :nslots])
                    input()
            if distill and self.history is not None and self.iter_cnt % self.period == 0:
                old_scores, old_inputs = self.forward(batch, nslots=self.history["nslots"], return_loss=False,
                                                      log_outputs=False, return_feature=True,
                                                      params=self.history["params"])
                self.iter_cnt -= 1
                old_scores = old_scores.detach()
                old_inputs = old_inputs.detach()
                new_scores = scores[:, :self.history["nslots"]]
                # new to old
                if mul_distill:
                    loss_distill = - torch.sum(
                        torch.softmax(old_scores * tau, dim=1) * torch.log_softmax(new_scores * tau, dim=1),
                        dim=1).mean()
                    old_dist = torch.softmax(old_scores / tau, dim=1)
                    old_valid = (old_dist[:, 0] < 0.9)
                    old_num = torch.sum(old_valid.float())
                    if old_num > 0:
                        loss_mul_distill = - torch.sum(
                            old_dist[old_valid] * torch.log_softmax(new_scores[old_valid], dim=1), dim=1).sum()
                        loss_distill = (loss_distill * old_dist.size(0) + loss_mul_distill) / (
                                    old_dist.size(0) + old_num)
                else:
                    loss_distill = - torch.sum(
                        torch.softmax(old_scores * tau, dim=1) * torch.log_softmax(new_scores * tau, dim=1),
                        dim=1).mean()
                if feature_distill:
                    loss_f_distill = (1 - (
                                old_inputs / old_inputs.norm(dim=-1, keepdim=True) * inputs / inputs.norm(dim=-1,
                                                                                                          keepdim=True)).sum(
                        dim=-1)).mean(dim=0)
                    loss_distill += loss_f_distill

                d_weight = self.history["nslots"]
                c_weight = (self.nslots - self.history["nslots"])
                loss = (d_weight * loss_distill + c_weight * loss) / (d_weight + c_weight)
                if torch.isnan(loss):
                    print(old_scores, new_scores)
                    input()
            if exemplar and self.exemplar_input is not None and self.iter_cnt % self.period == 0:
                idx = self.random_exemplar_inx[(int(self.iter_cnt/self.period) - 1) % self.exemplar_size]
                example_feature, exemplar_scores = self.replay_forward(params, task_id, idx)
                exemplar_scores[:, 0] = 0.
                label = self.exemplar_labels[idx].to(self.device).unsqueeze(0)
                loss_exemplar = self.crit(exemplar_scores + self.mask, label)
                if torch.isnan(loss_exemplar):
                    print(self.exemplar_labels, nslots)
                    input()
                if exemplar_distill:
                    example_old_feature, exemplar_old_scores = self.replay_forward(self.history["params"], task_id, idx)
                    exemplar_old_scores[:, 0] = 0.
                    exemplar_old_scores = exemplar_old_scores[:self.history["nslots"]]
                    loss_exemplar_distill = - torch.sum(
                        torch.softmax(exemplar_old_scores[:self.history["nslots"]] * tau, dim=1) * torch.log_softmax(
                            exemplar_scores[:self.history["nslots"]], dim=1), dim=1).mean()
                    # distill CLS token
                    if feature_distill:
                        loss_exemplar_feat_distill = (1 - (example_old_feature / example_old_feature.norm(dim=-1,
                                                                                                          keepdim=True) * example_feature / example_feature.norm(
                            dim=-1, keepdim=True)).sum(dim=-1)).mean(dim=0)
                        loss_exemplar_distill += loss_exemplar_feat_distill
                    d_weight = self.history["nslots"]
                    c_weight = (self.nslots - self.history["nslots"])
                    loss_exemplar = (d_weight * loss_exemplar_distill + c_weight * loss_exemplar) / (
                                d_weight + c_weight)

                loss = (nvalid * loss + self.e_weight * loss_exemplar) / (nvalid + self.e_weight)
                if torch.isnan(loss):
                    print(loss, loss_exemplar)
            return loss
        else:
            if return_feature:
                return scores[:, :nslots], inputs
            else:
                return scores[:, :nslots]

    def replay_forward(self, params, task_id, idx):

        attention_masks = self.exemplar_attm[idx].to(self.device)
        outputs = self.bert(self.exemplar_input[idx].to(self.device), attention_mask=attention_masks.unsqueeze(0))
        enc_outputs = outputs[0]
        bsz, seq_len, hidden_dim = enc_outputs.shape
        span_idx = self.exemplar_span[idx].to(self.device).unsqueeze(-1).unsqueeze(0)
        span_idx = span_idx.expand(-1, -1, hidden_dim)
        span_repr = torch.gather(enc_outputs, 1, span_idx)
        features = span_repr.view(1, hidden_dim * 2)
        inputs = self.input_map(features, params=self.get_subdict(params, "input_map"))
        example_feature = inputs
        scores = self.classes(inputs, params=self.get_subdict(params, "classes"))
        return example_feature, scores


    def score(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def clone_params(self, ):
        return OrderedDict({k: v.clone().detach() for k, v in self.meta_named_parameters()})

    def set_history(self, ):
        self.history = {"params": self.clone_params(), "nslots": self.nslots}

    def set_exemplar(self, dataloader, q: int = 20, params=None, label_sets: Union[List, Set, None] = None,
                     collect_none: bool = False, use_input: bool = False, output_only: bool = False,
                     output: Union[str, None] = None, task_id: int = 0):
        self.eval()
        with torch.no_grad():
            inid = []
            attm = []
            spans = []
            # pmt = []
            label = []
            ifeat = []
            ofeat = []
            example_batch = []
            num_batches = len(dataloader)
            for batch in dataloader:
                batch = batch.to(self.device)
                loss = self.forward(batch, params=params, store=True)
                for i in range(self.outputs["input"].size(0)):
                    inid.append(self.outputs["input"][i])
                    attm.append(self.outputs["attm"][i])
                ifeat.append(self.outputs["in_features"])
                spans.append(self.outputs["spans"])
                ofeat.append(self.outputs["encoded_features"])
                label.append(self.outputs["label"])
                example_batch.append(batch)

            spans = torch.cat(spans, dim=0)
            ifeat = torch.cat(ifeat, dim=0)
            ofeat = torch.cat(ofeat, dim=0)
            label = torch.cat(label, dim=0)
            nslots = max(self.nslots, torch.max(label).item() + 1)
            exemplar = {}
            if label_sets is None:
                if collect_none:
                    label_sets = range(nslots)
                else:
                    label_sets = range(1, nslots)
            else:
                if collect_none:
                    if 0 not in label_sets:
                        label_sets = sorted([0] + list(label_sets))
                    else:
                        label_sets = sorted(list(label_sets))
                else:
                    label_sets = sorted([t for t in label_sets if t != 0])
            for i in label_sets:
                idx = (label == i)
                if i == 0:
                    # random sample for none type
                    nidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
                    exemplar[i] = numpy.random.choice(nidx, q, replace=False).tolist()
                    continue
                if torch.any(idx):
                    exemplar[i] = []
                    nidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
                    mfeat = torch.mean(ofeat[idx], dim=0, keepdims=True)
                    if len(nidx) < q:
                        exemplar[i].extend(nidx * (q // len(nidx)) + nidx[:(q % len(nidx))])
                    else:
                        for j in range(q):
                            if j == 0:
                                dfeat = torch.sum((ofeat[nidx] - mfeat) ** 2, dim=1)
                            else:
                                cfeat = ofeat[exemplar[i]].sum(dim=0, keepdims=True)
                                cnum = len(exemplar[i])
                                dfeat = torch.sum((mfeat * (cnum + 1) - ofeat[nidx] - cfeat) ** 2, )
                            tfeat = torch.argmin(dfeat)
                            exemplar[i].append(nidx[tfeat])
                            nidx.pop(tfeat.item())
            exemplar = {i: ([inid[idx] for idx in v], [attm[idx] for idx in v], label[v],
                            spans[v], ifeat[v]) for i, v in exemplar.items()}
            exemplar_input = []
            exemplar_attm = []
            exemplar_span = []
            exemplar_features = []
            exemplar_labels = []
            for label, pack in exemplar.items():
                exemplar_input.extend(pack[0])
                exemplar_attm.extend(pack[1])
                exemplar_span.append(pack[3])
                exemplar_features.append(pack[4])
                exemplar_labels.extend([label] * pack[3].size(0))

            exemplar_span = torch.cat(exemplar_span, dim=0).cpu()
            exemplar_features = torch.cat(exemplar_features, dim=0).cpu()
            exemplar_labels = torch.LongTensor(exemplar_labels).cpu()
            if not output_only or output is not None:
                if output == "train" or output is None:
                    if self.exemplar_input is None:
                        self.exemplar_input = exemplar_input
                        self.exemplar_attm = exemplar_attm
                        self.exemplar_span = exemplar_span
                        self.exemplar_features = exemplar_features
                        self.exemplar_labels = exemplar_labels
                    else:
                        self.exemplar_input.extend(exemplar_input)
                        self.exemplar_attm.extend(exemplar_attm)
                        self.exemplar_span = torch.cat((self.exemplar_span, exemplar_span), dim=0)
                        self.exemplar_features = torch.cat((self.exemplar_features, exemplar_features), dim=0)
                        self.exemplar_labels = torch.cat((self.exemplar_labels, exemplar_labels), dim=0)
                elif output == "dev":
                    if self.dev_exemplar_features is None:
                        self.dev_exemplar_features = exemplar_features
                        self.dev_exemplar_labels = exemplar_labels
                    else:
                        self.dev_exemplar_features = torch.cat((self.dev_exemplar_features, exemplar_features), dim=0)
                        self.dev_exemplar_labels = torch.cat((self.dev_exemplar_labels, exemplar_labels), dim=0)

        exem_idx = list(range(self.exemplar_features.size(0)))
        random.shuffle(exem_idx)
        self.random_exemplar_inx = exem_idx
        self.exemplar_size = len(self.random_exemplar_inx)
        return {i: v[4].cpu() for i, v in exemplar.items()}


    def extra_forward(self, batch, nslots: int = -1, exemplar: bool = False, exemplar_distill: bool = False,
                feature_distill: bool = False, mul_distill=False, distill: bool = False, return_loss: bool = True,
                return_feature: bool = False, tau: float = 1.0, log_outputs: bool = True, params=None, task_id: int = 0,
                store: bool = False):

        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            features, labels = batch
        else:
            features, labels = batch.features, batch.labels

        inputs = self.input_map(features, params=self.get_subdict(params, "input_map"))
        scores = self.classes(inputs, params=self.get_subdict(params, "classes"))

        if torch.any(torch.isnan(scores)):
            print(scores[0])
            input('a')
        if nslots == -1:
            scores += self.mask
            if torch.any(torch.isnan(scores)):
                print(scores[0])
                input()
            nslots = self.nslots
        else:
            scores += self.idx_mask(max_idx=nslots)
        scores[:, 0] = 0

        if scores.size(0) != labels.size(0):
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        else:
            labels = labels
        if log_outputs:
            pred = torch.argmax(scores, dim=1)
            acc = torch.mean((pred == labels).float())
            self.outputs["accuracy"] = acc.item()
            self.outputs["prediction"] = pred.detach().cpu()

            # self.outputs["attm"] = span_attm.detach().cpu()
            self.outputs["label"] = labels.detach().cpu()
            self.outputs["in_features"] = features.detach().cpu()
            # self.outputs["prom"] = span_prom.detach().cpu()
            self.outputs["encoded_features"] = inputs.detach().cpu()
            # self.outputs["prompt"] = span_prompt_repr.detach().cpu()

        if return_loss:
            labels.masked_fill_(labels >= nslots, 0)
            valid = labels < nslots
            nvalid = torch.sum(valid.float())
            if nvalid == 0:
                loss = 0
            else:
                loss = self.crit(scores[valid], labels[valid])
                if torch.isnan(loss):
                    print(labels, nslots, scores[:, :nslots])
                    input()
            return loss
        else:
            if return_feature:
                return scores[:, :nslots], inputs
            else:
                return scores[:, :nslots]

    def set(self, features: torch.tensor, ids: Union[int, torch.Tensor, List, None] = None, max_id: int = -1):
        with torch.no_grad():
            if isinstance(ids, (torch.Tensor, list)):
                if torch.any(ids > self.nslots):
                    warnings.warn(
                        "Setting features to new classes. Using 'extend' or 'append' is preferred for new classes")
                self.classes.weight[ids] = features
            elif isinstance(ids, int):
                self.classes.weight[ids] = features
            else:
                if max_id == -1:
                    raise ValueError(f"Need input for either ids or max_id")
                self.classes.weight[:max_id] = features

    def append(self, feature):
        with torch.no_grad():
            self.classes.weight[self.nslots] = feature
            self.nslots += 1

    def extend(self, features):
        with torch.no_grad():
            features = features.to(self.device)
            if len(features.size()) == 1:
                warnings.warn("Extending 1-dim feature vector. Using 'append' instead is preferred.")
                self.append(features)
            else:
                nclasses = features.size(0)
                self.classes.weight[self.nslots:self.nslots + nclasses] = features
                self.nslots += nclasses


def test():  # sanity check
    m = KDR(nhead=8, nlayers=3, hidden_dim=512, input_dim=2048, max_slots=30, init_slots=9,
              device=torch.device("cpu"))


if __name__ == "__main__":
    test()
