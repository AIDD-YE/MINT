import numpy as np
import torch


def sanitize_errors(labels, scores, eps=1e-5):
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isnan(scores), 0.5, scores)
    scores = np.clip(scores, eps, 1.0 - eps)
    return scores


@torch.no_grad()
def predict_pocketnet(model, loader, device):
    model.eval()
    labels = []
    probabilities = []
    for tokens, mask, graph, chem, morgan, target, teacher in loader:
        tokens = tokens.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        graph = graph.to(device)
        if chem is not None:
            chem = chem.to(device, non_blocking=True)
        if morgan is not None:
            morgan = morgan.to(device, non_blocking=True)
        logits = model(tokens, mask, graph, chem, morgan)
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(labels, 0), np.concatenate(probabilities, 0)


@torch.no_grad()
def predict_mlp(model, loader, device):
    model.eval()
    labels = []
    probabilities = []
    for protein_vec, ligand_vec, target, teacher in loader:
        protein_vec = protein_vec.to(device, non_blocking=True)
        ligand_vec = ligand_vec.to(device, non_blocking=True)
        logits = model(protein_vec, ligand_vec)
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(labels, 0), np.concatenate(probabilities, 0)


def dispatch_predict(kind, model, loader, device):
    if kind == "pocketnet":
        return predict_pocketnet(model, loader, device)
    return predict_mlp(model, loader, device)