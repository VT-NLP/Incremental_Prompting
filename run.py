import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
from tqdm import tqdm
from utils.optimizer import AdamW
from utils.options import parse_arguments
from utils.dataloader import get_stage_loaders, get_stage_loaders_n
from utils.worker import Worker
from models.emp import PromptNet
from models.baseline import KDR
import random

PERM = [[0, 1, 2, 3,4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def by_class(preds, labels, learned_labels=None):
    match = (preds == labels).float()
    nlabels = max(torch.max(labels).item(), torch.max(preds).item())
    bc = {}

    ag = 0; ad = 0; am = 0
    for label in range(1, nlabels+1):
        lg = (labels==label); ld = (preds==label)
        lr = torch.sum(match[lg]) / torch.sum(lg.float())
        lp = torch.sum(match[ld]) / torch.sum(ld.float())
        lf = 2 * lr * lp / (lr + lp)
        if torch.isnan(lf):
            bc[label] = (0, 0, 0)
        else:
            bc[label] = (lp.item(), lr.item(), lf.item())
        if learned_labels is not None and label in learned_labels:
            ag += lg.float().sum()
            ad += ld.float().sum()
            am += match[lg].sum()
    if learned_labels is None:
        ag = (labels!=0); ad = (preds!=0)
        sum_ad = torch.sum(ag.float())
        if sum_ad == 0:
            ap = ar = 0
        else:
            ar = torch.sum(match[ag]) / torch.sum(ag.float())
            ap = torch.sum(match[ad]) / torch.sum(ad.float())
    else:
        if ad == 0:
            ap = ar = 0
        else:
            ar = am / ag; ap = am / ad
    if ap == 0:
        af = ap = ar = 0
    else:
        af = 2 * ar * ap / (ar + ap)
        af = af.item(); ar = ar.item(); ap = ap.item()
    return bc, (ap, ar, af)


def main():
    
    opts = parse_arguments()
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    summary = SummaryWriter(opts.log_dir)

    dataset_id = 0
    perm_id = opts.perm_id

    streams = json.load(open(opts.stream_file))
    streams = [streams[t] for t in PERM[perm_id]]
    loaders, exemplar_loaders, stage_labels, label2id = get_stage_loaders(root=opts.json_root,
        batch_size=opts.batch_size,
        streams=streams,
        num_workers=1,
        dataset=dataset_id)

    model = PromptNet(
        nhead=opts.nhead,
        nlayers=opts.nlayers,
        input_dim=opts.input_dim,
        hidden_dim=opts.hidden_dim,
        max_slots=opts.max_slots,
        init_slots=max(stage_labels[0])+1 if not opts.test_only else max(stage_labels[-1])+1,
        label_mapping=label2id,
        device=torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
    )
    param_groups = [
        {"params": [param for name, param in model.named_parameters() if param.requires_grad and 'correction' not in name],
        "lr":opts.learning_rate,
        "weight_decay": opts.decay,
        "betas": (0.9, 0.999)}
        ]
    optimizer = AdamW(params=param_groups)

    worker = Worker(opts)
    worker._log(str(opts))
    worker._log(str(label2id))
    if opts.test_only:
        worker.load(model, path=opts.model_dir)

    best_dev = best_test = None
    collect_stats = "accuracy"
    collect_outputs = {"prediction", "label"}
    termination = False
    patience = opts.patience
    no_better = 0
    loader_id = 0
    total_epoch = 0
    none_mul = 4
    learned_labels = set(stage_labels[0])
    best_dev_scores = []
    best_test_scores = []
    dev_metrics = None
    test_metrics = None
    exemplar_flag = True
    while not termination:
        if not opts.test_only:
            if opts.finetune:
                train_loss = lambda batch:model.forward(batch)
            elif opts.balance == "fd":
                train_loss = lambda batch:model.forward(batch, exemplar=True, feature_distill=True, exemplar_distill=True, distill=True, tau=0.5, task_id=loader_id)
            elif opts.balance == "mul":
                train_loss = lambda batch:model.forward(batch, exemplar=True, mul_distill=True, exemplar_distill=True, distill=True, tau=0.5, task_id=loader_id)
            else:
                train_loss = lambda batch:model.forward(batch, exemplar=exemplar_flag, exemplar_distill=True, distill=True, feature_distill=True, tau=0.5, task_id=loader_id)
            epoch_loss, epoch_metric = worker.run_one_epoch(
                model=model,
                f_loss=train_loss,
                loader=loaders[loader_id],
                split="train",
                optimizer=optimizer,
                collect_stats=collect_stats,
                prog=loader_id)
            total_epoch += 1
            # reset iter counter
            model.iter_cnt = 0
            # shuffle examplar index
            if loader_id > 0 and exemplar_flag:
                random.seed(opts.seed+99*total_epoch)
                random.shuffle(model.random_exemplar_inx)

            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}  Train Loss {epoch_loss} {epoch_metric}")
        else:
            learned_labels = set([t for stream in stage_labels for t in stream])
            termination = True

        if opts.test_only:
            score_fn = model.score
            test_loss, test_metrics = worker.run_one_epoch(
                model=model,
                f_loss=score_fn,
                loader=loaders[-1],
                split="test",
                collect_stats=collect_stats,
                collect_outputs=collect_outputs)
            test_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
            torch.save(test_outputs, f"log/{os.path.basename(opts.load_model)}.output")
            test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"], learned_labels=learned_labels)
            test_class_f1 = {k: test_scores[k][2] for k in test_scores}
            for k,v in test_class_f1.items():
                add_summary_value(summary, f"test_class_{k}", v, total_epoch)
            test_metrics = test_f
            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}: Test {test_metrics}"
                )
        if not opts.test_only:
            score_fn = model.score
            dev_loss, dev_metrics = worker.run_one_epoch(
                    model=model,
                    f_loss=score_fn,
                    loader=loaders[-2],
                    split="dev",
                    collect_stats=collect_stats,
                    collect_outputs=collect_outputs)
            dev_outputs = {k: torch.cat(v, dim=0) for k, v in worker.epoch_outputs.items()}
            dev_scores, (dev_p, dev_r, dev_f) = by_class(dev_outputs["prediction"], dev_outputs["label"],
                                                             learned_labels=learned_labels)
            dev_class_f1 = {k: dev_scores[k][2] for k in dev_scores}
            for k, v in dev_class_f1.items():
                add_summary_value(summary, f"dev_class_{k}", v, total_epoch)
            dev_metrics = dev_f
            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}:  Dev {dev_metrics}"
                )
            if best_dev is None or dev_metrics > best_dev:
                print("-----find best model on dev-----")
                best_dev = dev_metrics
                worker.save(model, optimizer, postfix=str(loader_id))   # save best model on dev
                # whether reset patient when a better dev found
            else:
                no_better += 1
                print("-----hit patience-----")
            print(f"patience: {no_better} / {patience}")

            if (no_better == patience) or (worker.epoch == worker.train_epoch):
                if no_better == patience:
                    print("------early stop-----")

                loader_id += 1
                no_better = 0
                worker.load(model, optimizer, path=os.path.join(opts.log_dir, f"{worker.save_model}.{loader_id - 1}"))

                test_loss, test_metrics = worker.run_one_epoch(
                    model=model,
                    f_loss=score_fn,
                    loader=loaders[-1],
                    split="test",
                    collect_stats=collect_stats,
                    collect_outputs=collect_outputs)
                test_outputs = {k: torch.cat(v, dim=0) for k, v in worker.epoch_outputs.items()}
                torch.save(test_outputs, f"./log/{os.path.basename(opts.load_model)}.output")
                test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"],
                                                                 learned_labels=learned_labels)
                test_class_f1 = {k: test_scores[k][2] for k in test_scores}
                for k, v in test_class_f1.items():
                    add_summary_value(summary, f"test_class_{k}", v, total_epoch)

                test_metrics = test_f
                best_test = test_metrics
                print("-----Test F1-----")
                print(best_test)

                best_dev_scores.append(best_dev)
                best_test_scores.append(best_test)
                print("-----------Current Best Dev Results----------")
                print(best_dev_scores)
                print("-----------Current Best Test Results----------")
                print(best_test_scores)

                if not opts.finetune:
                    print("setting train exemplar for learned classes")
                    model.set_exemplar(exemplar_loaders[loader_id-1], task_id=loader_id-1)

                # set prompt's require_grad
                if loader_id == 1:
                    model.prompted_embed.prompt2.requires_grad = True
                elif loader_id == 2:
                    model.prompted_embed.prompt3.requires_grad = True
                elif loader_id == 3:
                    model.prompted_embed.prompt4.requires_grad = True
                elif loader_id == 4:
                    model.prompted_embed.prompt5.requires_grad = True

                if not opts.finetune:
                    model.set_history()
                for output_log in [print, worker._log]:
                    output_log(f"BEST DEV {loader_id-1}: {best_dev if best_dev is not None else 0}")
                    output_log(f"BEST TEST {loader_id-1}: {best_test if best_test is not None else 0}")
                if loader_id == len(loaders) - 2:
                    termination = True
                else:
                    learned_labels = learned_labels.union(set(stage_labels[loader_id]))
                    model.nslots = max(learned_labels) + 1
                worker.epoch = 0
                best_dev = None; best_test = None
    print("-----------Dev Results----------")
    print(best_dev_scores)
    print("-----------Test Results----------")
    print(best_test_scores)


if __name__ == "__main__":
    main()
