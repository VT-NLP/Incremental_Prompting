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
# task split for MAVEN
if opts.perm_id == 0:
    TASK_EVENT_NUM = [33, 30, 39, 35, 31]
    NA_TASK_EVENT_NUM = [1, 33, 30, 39, 35, 31]
elif opts.perm_id == 1:
    TASK_EVENT_NUM = [31, 35, 39, 30, 33]
    NA_TASK_EVENT_NUM = [1, 31, 35, 39, 30, 33]
elif opts.perm_id == 2:
    TASK_EVENT_NUM = [33, 35, 30, 31, 39]
    NA_TASK_EVENT_NUM = [1, 33, 35, 30, 31, 39]
elif opts.perm_id == 3:
    TASK_EVENT_NUM = [30, 39, 33, 35, 31]
    NA_TASK_EVENT_NUM = [1, 30, 39, 33, 35, 31]
elif opts.perm_id == 4:
    TASK_EVENT_NUM = [35, 31, 33, 30, 39]
    NA_TASK_EVENT_NUM = [1, 35, 31, 33, 30, 39]

# task split for ACE
# if opts.perm_id == 0:
#     TASK_EVENT_NUM = [9, 6, 5, 5, 8]
#     NA_TASK_EVENT_NUM = [1, 9, 6, 5, 5, 8]
# elif opts.perm_id == 1:
#     TASK_EVENT_NUM = [8, 5, 5, 6, 9]
#     NA_TASK_EVENT_NUM = [1, 8, 5, 5, 6, 9]
# elif opts.perm_id == 2:
#     TASK_EVENT_NUM = [9, 5, 6, 8, 5]
#     NA_TASK_EVENT_NUM = [1, 9, 5, 6, 8, 5]
# elif opts.perm_id == 3:
#     TASK_EVENT_NUM = [6, 5, 9, 5, 8]
#     NA_TASK_EVENT_NUM = [1, 6, 5, 9, 5, 8]
# elif opts.perm_id == 4:
#     TASK_EVENT_NUM = [5, 8, 9, 6, 5]
#     NA_TASK_EVENT_NUM = [1, 5, 8, 9, 6, 5]

PROMPT_SIZE = opts.prompt_size
device = torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
# file_path = "./data/ACE/id2tokens.json"
file_path = "./data/MAVEN/id2tokens_unk.json"
random.seed(opts.seed)


class SoftPrompt(nn.Module):
    def __init__(self, plm_embed: nn.Embedding, label_mapping: dict, n_prompt: int = PROMPT_SIZE, random_range: float = 0.1,
                 initialize_from_vocab: bool = False):
        super(SoftPrompt, self).__init__()
        self.plm_embed = plm_embed
        self.n_prompt = n_prompt
        self.id2label = {}
        for label in label_mapping.keys():
            self.id2label[label_mapping[label]] = label
        self.embed_size = plm_embed.weight.size(1)
        self.random_range = random_range
        self.event_tensors = self.build_event_repr(self.random_range)

        self.event_tensor_list = self.sep_event_tensor()
        self.prompt_na = nn.parameter.Parameter(self.event_tensor_list[0], requires_grad=True)
        self.prompt1 = nn.parameter.Parameter(self.event_tensor_list[1], requires_grad=True)
        self.prompt2 = nn.parameter.Parameter(self.event_tensor_list[2], requires_grad=False)
        self.prompt3 = nn.parameter.Parameter(self.event_tensor_list[3], requires_grad=False)
        self.prompt4 = nn.parameter.Parameter(self.event_tensor_list[4], requires_grad=False)
        self.prompt5 = nn.parameter.Parameter(self.event_tensor_list[5], requires_grad=False)
        self.prompt_sep = nn.parameter.Parameter(self.event_tensor_list[6], requires_grad=True)

    def build_event_repr(self, random_range):
        init_tensors = []
        with open(file_path, 'rt') as fp:
            id2tokens = json.load(fp)
        for i in range(len(id2tokens)):

            event_token_ids = id2tokens[str(self.id2label[i])]
            token_len = len(event_token_ids)
            _temp = torch.zeros(self.embed_size, dtype=torch.float)
            for event_token_id in event_token_ids:
                if event_token_id == 100:
                    _temp = torch.FloatTensor(self.embed_size).uniform_(-random_range, random_range)
                    break
                else:
                    _temp += self.plm_embed.weight[event_token_id].clone().detach()
            init_tensors.append(_temp / token_len)

        init_tensors.append(self.plm_embed.weight[102].clone().detach())
        return torch.stack(init_tensors, dim=0)

    def sep_event_tensor(self):
        tensor_list = []
        j = 0
        for i in NA_TASK_EVENT_NUM:
            event_tensor = self.event_tensors[j:(i+j), :]
            tensor_list.append(event_tensor)
            j = i + j
        sep = self.event_tensors[-1, :].unsqueeze(0)
        tensor_list.append(sep)
        return tensor_list

    def forward(self, tokens):
        prompt_embed = torch.cat([self.prompt_na, self.prompt1, self.prompt2, self.prompt3, self.prompt4, self.prompt5, self.prompt_sep], dim=0)
        input_embedding = self.plm_embed(tokens[:, :-self.n_prompt])
        bsz, seq_len, hidden_dim = input_embedding.shape
        prompt_embed = prompt_embed.repeat(bsz, 1, 1)
        return torch.cat([input_embedding, prompt_embed], 1)


class PromptNet(MetaModule):
    def __init__(self, input_dim: int, hidden_dim: int, max_slots: int, init_slots: int, label_mapping: dict,
                 device: Union[torch.device, None] = None, **kwargs) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-large-cased')
        for name, param in list(self.bert.named_parameters()):
            param.requires_grad = False
        self.prompted_embed = SoftPrompt(plm_embed=self.bert.get_input_embeddings(), label_mapping=label_mapping,
                                    n_prompt=PROMPT_SIZE+1)

        self.bert.set_input_embeddings(self.prompted_embed)
        self.half_input_dim = int(input_dim / 2)
        self.input_map = MetaSequential(OrderedDict({
            "linear_0": MetaLinear(input_dim, hidden_dim),
            "relu_0": nn.ReLU(),
            "dropout_0": nn.Dropout(0.2),
            "linear_1": MetaLinear(hidden_dim, hidden_dim),
            "relu_1": nn.ReLU(),
            "dropout_1": nn.Dropout(0.2),
            "linear_2": MetaLinear(hidden_dim, hidden_dim),
            "relu_2": nn.ReLU()
        }))
        self.prompt_map = MetaSequential(OrderedDict({
            "linear_0": MetaLinear(self.half_input_dim, hidden_dim),
            "relu_0": nn.ReLU(),
            "linear_1": MetaLinear(hidden_dim, hidden_dim),
            "relu_1": nn.ReLU(),
            "linear_2": MetaLinear(hidden_dim, hidden_dim),
            "relu_2": nn.ReLU()
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
        self.outputs = {"input_ids": []}
        self.history = None
        self.exemplar_input = None
        self.exemplar_attm = None
        self.exemplar_labels = None
        self.exemplar_span = None
        self.exemplar_logit = None
        self.exemplar_feature = None
        self.exemplar_prompt = None
        self.exemplar_enc_prompt = None
        self.exemplar_size = None
        self.random_exemplar_inx = None

        prompt_mask_list = self.get_prompt_mask()
        self.prompt_mask_list = torch.stack(prompt_mask_list, dim=0).to(device)
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

    def get_prompt_mask(self):
        prompt_list = []
        j = 1
        for i in TASK_EVENT_NUM:
            prompt_zero_mask = torch.zeros(self.max_slots, self.half_input_dim)
            prompt_zero_mask[0:(i+j), :] = 1
            prompt_list.append(prompt_zero_mask)
            j = i + j
        return prompt_list

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
        self.outputs["input_ids"].append(input_ids)

        attention_masks = attention_masks[task_id]

        outputs = self.bert(input_ids.unsqueeze(0), attention_mask=attention_masks.unsqueeze(0))
        enc_outputs = outputs[0]
        bsz, seq_len, hidden_dim = enc_outputs.shape
        rep_enc_outputs = enc_outputs.repeat(span_bsz, 1, 1)

        span_idx = spans.unsqueeze(-1)
        span_idx = span_idx.expand(-1, -1, hidden_dim)
        span_repr = torch.gather(rep_enc_outputs, 1, span_idx)
        # input feature: (span_bsz, hidden_dim*2)
        features = span_repr.view(span_bsz, hidden_dim*2)

        prompt_idx = prompt_masks.unsqueeze(-1).unsqueeze(0)
        prompt_idx = prompt_idx.expand(-1, -1, hidden_dim)
        # prompt hidden: (bsz, num_prompt, hidden_dim)
        prompt_repr = torch.gather(enc_outputs, 1, prompt_idx)
        pro_mask = self.prompt_mask_list[task_id, :].unsqueeze(0)
        prompt_repr = prompt_repr.mul(pro_mask)

        # original scores
        inputs = self.input_map(features, params=self.get_subdict(params, "input_map"))
        scores_s = self.classes(inputs, params=self.get_subdict(params, "classes"))
        prompt_classifier = self.prompt_map(prompt_repr, params=self.get_subdict(params, "prompt_map"))
        prompt_classifier = prompt_classifier.repeat(span_bsz, 1, 1)
        inputs_p = inputs.unsqueeze(-1)
        scores_p = torch.bmm(prompt_classifier, inputs_p).squeeze(-1)
        scores = scores_s + scores_p

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
            self.outputs["logit"] = scores.detach().cpu()
            self.outputs["label"] = labels.detach().cpu()
            self.outputs["spans"] = spans.detach().cpu()
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
                                                      params=self.history["params"], task_id=task_id)
                self.iter_cnt -= 1
                old_scores = old_scores.detach()
                old_inputs = old_inputs.detach()
                new_scores = scores[:, :self.history["nslots"]]
                if mul_distill:
                    loss_distill = - torch.sum(
                        torch.softmax(old_scores * tau, dim=1) * torch.log_softmax(new_scores * tau, dim=1),
                        dim=1).mean()
                    old_dist = torch.softmax(old_scores / tau, dim=1)
                    old_valid = (old_dist[:, 0] < 0.9)
                    old_num = torch.sum(old_valid.float())
                    if old_num > 0:
                        # print(old_dist[old_valid].topk(5, dim=1), batch.labels[old_valid])
                        # input()
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
                    exemplar_old_scores = self.exemplar_logit[idx].unsqueeze(0).to(self.device)
                    example_old_feature = self.exemplar_feature[idx].unsqueeze(0).to(self.device)
                    exemplar_old_scores[:, 0] = 0.
                    exemplar_old_scores = exemplar_old_scores[:self.history["nslots"]]
                    loss_exemplar_distill = - torch.sum(
                        torch.softmax(exemplar_old_scores[:self.history["nslots"]] * tau, dim=1) * torch.log_softmax(
                            exemplar_scores[:self.history["nslots"]], dim=1), dim=1).mean()
                    if feature_distill:
                        loss_exemplar_feat_distill = (1 - (example_old_feature / example_old_feature.norm(dim=-1,
                                                                                                          keepdim=True) * example_feature / example_feature.norm(
                            dim=-1, keepdim=True)).sum(dim=-1)).mean(dim=0)
                        loss_exemplar_distill += loss_exemplar_feat_distill
                    d_weight = self.history["nslots"]
                    c_weight = (self.nslots - self.history["nslots"]) * 10
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
        attention_masks = self.exemplar_attm[idx][task_id].to(self.device)
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



    def update_exem_feat(self, task_id):
        for i in range(self.exemplar_size):
            attention_masks = self.exemplar_attm[i][task_id].to(self.device)
            outputs = self.bert(self.exemplar_input[i].to(self.device), attention_mask=attention_masks.unsqueeze(0))
            self.exemplar_cls[i] = outputs[1].detach().cpu()

    def score(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def clone_params(self, ):
        return OrderedDict({k: v.clone().detach() for k, v in self.meta_named_parameters()})

    def set_history(self, ):
        self.history = {"params": self.clone_params(), "nslots": self.nslots}

    def set_exemplar(self, dataloader, q: int = 20, params=None, label_sets: Union[List, Set, None] = None,
                     collect_none: bool = False, use_input: bool = False, output_only: bool = False,
                     output: Union[str, None] = None, task_id:int = 0):
        self.eval()
        with torch.no_grad():
            inid = []
            attm = []
            spans = []
            pmt = []
            label = []
            ofeat = []
            example_batch = []

            for batch in dataloader:
                batch = batch.to(self.device)
                loss = self.forward(batch, params=params, store=True)
                for i in range(self.outputs["input"].size(0)):
                    inid.append(self.outputs["input"][i])
                    attm.append(self.outputs["attm"][i])
                    pmt.append(self.outputs["prom"])
                spans.append(self.outputs["spans"])
                ofeat.append(self.outputs["encoded_features"])
                label.append(self.outputs["label"])
                example_batch.append(batch)

            spans = torch.cat(spans, dim=0)
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
            exemplar = {i: ([inid[idx] for idx in v], [attm[idx] for idx in v], [pmt[idx] for idx in v], label[v],
                            spans[v]) for i, v in exemplar.items()}
            exemplar_input = []
            exemplar_attm = []
            exemplar_prompt = []
            exemplar_span = []
            exemplar_labels = []

            for label, pack in exemplar.items():
                exemplar_input.extend(pack[0])
                exemplar_attm.extend(pack[1])
                exemplar_prompt.extend(pack[2])
                exemplar_span.append(pack[4])
                exemplar_labels.extend([label] * pack[4].size(0))

            exemplar_span = torch.cat(exemplar_span, dim=0).cpu()
            exemplar_labels = torch.LongTensor(exemplar_labels).cpu()

            if not output_only or output is not None:
                if output == "train" or output is None:
                    if self.exemplar_input is None:
                        self.exemplar_input = exemplar_input
                        self.exemplar_attm = exemplar_attm
                        self.exemplar_span = exemplar_span
                        self.exemplar_prompt = exemplar_prompt
                        self.exemplar_labels = exemplar_labels

                    else:
                        self.exemplar_input.extend(exemplar_input)
                        self.exemplar_attm.extend(exemplar_attm)
                        self.exemplar_prompt.extend(exemplar_prompt)
                        self.exemplar_span = torch.cat((self.exemplar_span, exemplar_span), dim=0)
                        self.exemplar_labels = torch.cat((self.exemplar_labels, exemplar_labels), dim=0)

        exem_idx = list(range(self.exemplar_span.size(0)))
        random.shuffle(exem_idx)
        self.random_exemplar_inx = exem_idx
        self.exemplar_size = len(self.random_exemplar_inx)
        self.update_exemplar_feat(task_id, params)
        return {i: (v[0], v[1], v[2], v[3].cpu(), v[4].cpu()) for i, v in exemplar.items()}

    def update_exemplar_feat(self, task_id, params):
        print("---updating exemplar prompts---")
        exem_scores = []
        exem_feats = []
        with torch.no_grad():
            for i in range(self.exemplar_size):
                exem_feat, exem_score = self.replay_forward(params=params, task_id=task_id, idx=i)
                exem_feats.append(exem_feat.detach().cpu())
                exem_scores.append(exem_score.detach().cpu())
            self.exemplar_feature = torch.cat(exem_feats, dim=0)
            self.exemplar_logit = torch.cat(exem_scores, dim=0)


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
    m = PromptNet(nhead=8, nlayers=3, hidden_dim=512, input_dim=2048, max_slots=30, init_slots=9,
              device=torch.device("cpu"))


if __name__ == "__main__":
    test()
