from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, accuracy_score)

from data_loader import LEVEL_EDGES, build_full_matrix_loader
from model import LEVELS


def get_ec_pos_dict(mlb, true_label, pred_label):
    """Column index of every EC token in the true/predicted sets (from CLEAN's evaluate.py)."""
    ec_list, pos_list = [], []
    for i in range(len(true_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([true_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([true_label[i]]))[1])
    for i in range(len(pred_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([pred_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([pred_label[i]]))[1])
    return {ec_list[i]: pos_list[i] for i in range(len(ec_list))}


def clean_eval_metrics(pred_label, pred_probs, true_label, all_label):
    """Weighted multilabel P/R/F1 + sparse weighted ROC-AUC + subset accuracy (CLEAN's get_eval_metrics)."""
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    classes = set(mlb.classes_)
    n = len(pred_label)
    pred_m = np.zeros((n, len(mlb.classes_)))
    true_m = np.zeros((n, len(mlb.classes_)))
    pred_m_auc = np.zeros((n, len(mlb.classes_)))
    label_pos = get_ec_pos_dict(mlb, true_label, pred_label)
    for i in range(n):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
        for label, prob in zip(pred_label[i], pred_probs[i]):
            if label in classes:
                pred_m_auc[i, label_pos[label]] = prob
    pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted', zero_division=0)
    f1 = f1_score(true_m, pred_m, average='weighted', zero_division=0)
    try:
        roc = roc_auc_score(true_m, pred_m_auc, average='weighted')
    except ValueError:
        roc = float('nan')
    acc = accuracy_score(true_m, pred_m)
    return {'precision': pre, 'recall': rec, 'f1': f1, 'roc_auc': roc, 'accuracy': acc}


def eval_level(scores_by_protein, true_by_protein, prot_order, threshold):
    """CLEAN weighted metrics for one level"""
    prots = list(prot_order)
    universe = sorted(set().union(*[set(true_by_protein[p]) for p in prots])) if prots else []
    universe_set = set(universe)
    if not prots or not universe:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': float('nan')}

    true_label, pred_label, pred_probs = [], [], []
    for p in prots:
        true_label.append(list(true_by_protein[p]))
        toks, probs = [], []
        for tok, s in scores_by_protein[p].items():
            if tok in universe_set and s > threshold:
                toks.append(tok)
                probs.append(s)
        pred_label.append(toks)
        pred_probs.append(probs)
    m = clean_eval_metrics(pred_label, pred_probs, true_label, universe)
    return {'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'], 'roc_auc': m['roc_auc']}


@torch.no_grad()
def score_full_matrix(model, data, level, config, protein_subset, device, num_workers=2, desc='Scoring'):
    """Score the full (protein_subset x all nodes) matrix at `level`."""
    edge_type, _ = LEVEL_EDGES[level]
    loader, global_prots, global_nodes = build_full_matrix_loader(
        data, level, config, protein_subset, num_workers=num_workers)

    eval_device = torch.device('cpu') if config.get('test_on_cpu', True) else device
    model.to(eval_device)
    model.eval()

    all_probs, all_labels = [], []
    for batch in tqdm(loader, total=len(loader), desc=desc):
        batch = batch.to(eval_device)
        pred, _ = model(batch.x_dict, batch.edge_index_dict, batch[edge_type].edge_label_index, level)
        all_probs.append(torch.sigmoid(pred.cpu()).numpy())
        all_labels.append(batch[edge_type].edge_label.cpu().numpy())

    model.to(device)
    return np.concatenate(all_probs), np.concatenate(all_labels), global_prots, global_nodes


def score_all_levels(model, data, config, device, protein_subset, num_workers=2, desc_prefix='Eval'):
    """Score every level's full matrix once, returning {level: (scores_by_protein, true_by_protein, eval_prots)}."""
    scored = {}
    for level in LEVELS:
        probs, labels, gp, gn = score_full_matrix(
            model, data, level, config, protein_subset, device,
            num_workers=num_workers, desc=f'{desc_prefix} {level}')
        scores_by_protein, true_by_protein = defaultdict(dict), defaultdict(set)
        for p, n, pr, lb in zip(gp.tolist(), gn.tolist(), probs.tolist(), labels.tolist()):
            scores_by_protein[p][n] = pr
            if lb == 1:
                true_by_protein[p].add(n)
        eval_prots = [p for p in scores_by_protein if true_by_protein[p]]
        scored[level] = (scores_by_protein, true_by_protein, eval_prots)
    return scored


def metrics_from_scores(scored, threshold):
    """Per-level metrics from pre-scored levels, at a single probability threshold."""
    return {level: eval_level(sbp, tbp, prots, threshold)
            for level, (sbp, tbp, prots) in scored.items()}


def run_level_metrics(model, data, config, device, protein_subset, threshold,
                      num_workers=2, desc_prefix='Eval'):
    """Score all levels then compute per-level metrics at `threshold`."""
    scored = score_all_levels(model, data, config, device, protein_subset,
                              num_workers=num_workers, desc_prefix=desc_prefix)
    return metrics_from_scores(scored, threshold)
