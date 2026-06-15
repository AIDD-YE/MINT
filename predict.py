import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FEATURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.csv")

METHOD_CONFIGS = {
    "MINT":        {"model_type": "pocketnet", "dir": "MINT"},
    "MINT_wo_A":   {"model_type": "pocketnet", "dir": "MINT_wo_A"},
    "MINT_wo_D":   {"model_type": "pocketnet", "dir": "MINT_wo_D"},
    "Baseline":    {"model_type": "mlp",       "dir": "Baseline"},
    "Baseline_wo_A": {"model_type": "mlp",     "dir": "Baseline_wo_A"},
    "Baseline_wo_D": {"model_type": "mlp",     "dir": "Baseline_wo_D"},
}

N_FOLDS = 5


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def masked_mean(x, mask, dim=1, eps=1e-6):
    w = mask.float().unsqueeze(-1)
    return (x * w).sum(dim=dim) / (w.sum(dim=dim) + eps)


def preload_features(prot_dir, lig_dir, needed_prot_keys, needed_lig_keys):
    prot_cache = {}
    for k in needed_prot_keys:
        p = os.path.join(prot_dir, f"{k}.pkl")
        if os.path.exists(p):
            prot_cache[k] = load_pkl(p)
    lig_cache = {}
    for k in needed_lig_keys:
        p = os.path.join(lig_dir, f"{k}.pkl")
        if os.path.exists(p):
            lig_cache[k] = load_pkl(p)
    return prot_cache, lig_cache


class ProteinEncoder(nn.Module):
    def __init__(self, d_in=1280, d_model=256, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, P, Pmask):
        x = self.proj(P)
        x = self.enc(x, src_key_padding_mask=~Pmask)
        return self.ln(x)


class LigandGIN(nn.Module):
    def __init__(self, d_in=43, d_model=256, n_layers=4, dropout=0.1):
        super().__init__()
        self.x_proj = nn.Linear(d_in, d_model)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            self.convs.append(GINConv(mlp))
        self.lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = dropout

    def forward(self, G: Batch):
        h = self.x_proj(G.x)
        for conv, ln in zip(self.convs, self.lns):
            h2 = conv(h, G.edge_index)
            h2 = F.gelu(h2)
            h2 = F.dropout(h2, p=self.dropout, training=self.training)
            h = ln(h + h2)
        g = global_mean_pool(h, G.batch)
        return h, g


class CrossAttnBinder(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1, use_globals=True):
        super().__init__()
        self.use_globals = use_globals
        self.prot_enc = ProteinEncoder(d_in=1280, d_model=d_model, n_layers=2, n_heads=n_heads, dropout=dropout)
        self.lig_enc = LigandGIN(d_in=43, d_model=d_model, n_layers=4, dropout=dropout)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        if use_globals:
            self.chem_proj = nn.Linear(768, d_model)
            self.morgan_proj = nn.Linear(2048, d_model)
        in_dim = d_model * (4 + (2 if use_globals else 0))
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, P, Pmask, G: Batch, CHEM=None, MORGAN=None):
        p_tok = self.prot_enc(P, Pmask)
        a_node, a_graph = self.lig_enc(G)
        device = P.device
        B = int(G.batch.max().item()) + 1
        d = a_node.size(-1)
        counts = torch.bincount(G.batch, minlength=B).tolist()
        Nmax = max(counts) if len(counts) else 0
        A = a_node.new_zeros((B, Nmax, d))
        Amask = torch.zeros((B, Nmax), dtype=torch.bool, device=device)
        start = 0
        for i, n in enumerate(counts):
            if n > 0:
                A[i, :n] = a_node[start:start + n]
                Amask[i, :n] = True
                start += n
        attn_out, _ = self.cross(p_tok, A, A, key_padding_mask=~Amask, need_weights=False)
        p0 = masked_mean(p_tok, Pmask)
        a0 = a_graph
        pa = masked_mean(attn_out, Pmask)
        feats = [p0, a0, pa, p0 * a0]
        if self.use_globals and (CHEM is not None) and (MORGAN is not None):
            feats.append(self.chem_proj(CHEM))
            feats.append(self.morgan_proj(MORGAN))
        x = torch.cat(feats, dim=-1)
        return self.head(x).squeeze(1)


class SimpleMLP(nn.Module):
    def __init__(self, d_prot=1280, d_lig=768, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_prot + d_lig, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, prot_vec, lig_vec):
        x = torch.cat([prot_vec, lig_vec], dim=1)
        return self.net(x).squeeze(1)


class PocketNetDataset(Dataset):
    def __init__(self, df, prot_cache, lig_cache):
        self.df = df.reset_index(drop=True)
        self.prot_cache = prot_cache
        self.lig_cache = lig_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[int(i)]
        prot_key = str(row["prot_key"])
        lig_key = str(row["lig_key"])
        prot = self.prot_cache[prot_key]
        lig = self.lig_cache[lig_key]
        p_tok = torch.from_numpy(prot["token_representation"].astype(np.float32))
        x = lig["atom_feature"].float()
        edge_index = lig["edge_index"].long()
        data = Data(x=x, edge_index=edge_index)
        edge_weight = lig.get("edge_weight", None)
        if edge_weight is not None:
            data.edge_weight = edge_weight.float()
        chem = lig.get("chemBERTa", None)
        morgan = lig.get("morgan_fingerprint", None)
        chem = chem.float().view(-1) if chem is not None else torch.zeros(768)
        morgan = morgan.float().view(-1) if morgan is not None else torch.zeros(2048)
        return p_tok, data, chem, morgan


def collate_pocketnet(batch):
    p_tok, g, chem, morgan = zip(*batch)
    B = len(p_tok)
    d_in = p_tok[0].shape[1]
    Lmax = max(t.shape[0] for t in p_tok)
    P = torch.zeros((B, Lmax, d_in), dtype=torch.float32)
    Pmask = torch.zeros((B, Lmax), dtype=torch.bool)
    for i, t in enumerate(p_tok):
        L = t.shape[0]
        P[i, :L] = t
        Pmask[i, :L] = True
    G = Batch.from_data_list(list(g))
    CHEM = torch.stack(list(chem), 0)
    MORGAN = torch.stack(list(morgan), 0)
    return P, Pmask, G, CHEM, MORGAN


class MLPDataset(Dataset):
    def __init__(self, df, prot_cache, lig_cache):
        self.df = df.reset_index(drop=True)
        self.prot_cache = prot_cache
        self.lig_cache = lig_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[int(i)]
        prot_key = str(row["prot_key"])
        lig_key = str(row["lig_key"])
        prot = self.prot_cache[prot_key]
        lig = self.lig_cache[lig_key]
        tok = prot["token_representation"].astype(np.float32)
        pvec = torch.from_numpy(tok).mean(dim=0)
        c = lig.get("chemBERTa", None)
        lvec = c.float().view(-1)
        return pvec, lvec


def collate_mlp(batch):
    pvec, lvec = zip(*batch)
    return torch.stack(pvec, 0), torch.stack(lvec, 0)


@torch.no_grad()
def predict_pocketnet(model, loader):
    model.eval()
    probs = []
    for P, Pmask, G, CHEM, MORGAN in loader:
        P = P.to(DEVICE, non_blocking=True)
        Pmask = Pmask.to(DEVICE, non_blocking=True)
        G = G.to(DEVICE)
        CHEM = CHEM.to(DEVICE, non_blocking=True)
        MORGAN = MORGAN.to(DEVICE, non_blocking=True)
        logit = model(P, Pmask, G, CHEM, MORGAN)
        prob = torch.sigmoid(logit).detach().cpu().numpy()
        probs.append(prob)
    return np.concatenate(probs, 0)


@torch.no_grad()
def predict_mlp(model, loader):
    model.eval()
    probs = []
    for pvec, lvec in loader:
        pvec = pvec.to(DEVICE, non_blocking=True)
        lvec = lvec.to(DEVICE, non_blocking=True)
        logit = model(pvec, lvec)
        prob = torch.sigmoid(logit).detach().cpu().numpy()
        probs.append(prob)
    return np.concatenate(probs, 0)


def predict_method(method_name, config, test_df, prot_cache, lig_cache):
    model_type = config["model_type"]
    ckpt_dir = os.path.join(MODELS_DIR, config["dir"])
    fold_probs = []
    for fold_idx in range(1, N_FOLDS + 1):
        if model_type == "pocketnet":
            model = CrossAttnBinder(d_model=256, n_heads=4, dropout=0.1, use_globals=True).to(DEVICE)
            ds = PocketNetDataset(test_df, prot_cache, lig_cache)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pocketnet)
            pred_fn = predict_pocketnet
        else:
            model = SimpleMLP(d_prot=1280, d_lig=768, hidden=512, dropout=0.1).to(DEVICE)
            ds = MLPDataset(test_df, prot_cache, lig_cache)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mlp)
            pred_fn = predict_mlp
        ckpt_path = os.path.join(ckpt_dir, f"fold{fold_idx}", "model.pt")
        payload = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["model_state"])
        prob = pred_fn(model, loader)
        fold_probs.append(prob)
        print(f"  {method_name} fold{fold_idx} done", flush=True)
    ensemble = np.mean(np.stack(fold_probs, 0), 0)
    return ensemble


def main():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_manifest.csv"))
    print(f"Test samples: {len(test_df)}", flush=True)

    prot_keys = set(test_df["prot_key"].astype(str).tolist())
    lig_keys = set(test_df["lig_key"].astype(str).tolist())
    prot_dir = os.path.join(FEATURES_DIR, "protein")
    lig_dir = os.path.join(FEATURES_DIR, "ligand")
    prot_cache, lig_cache = preload_features(prot_dir, lig_dir, prot_keys, lig_keys)
    print(f"Loaded prot features: {len(prot_cache)}, lig features: {len(lig_cache)}", flush=True)

    out = test_df[["id", "true_id", "ligand_code", "pdb_code", "classification_label"]].copy()

    for method_name, config in METHOD_CONFIGS.items():
        print(f"Predicting {method_name}...", flush=True)
        probs = predict_method(method_name, config, test_df, prot_cache, lig_cache)
        out[method_name] = probs.astype(np.float32)

    plxfpred_df = pd.read_csv(os.path.join(DATA_DIR, "plxfpred_test_predictions.csv"))
    plxfpred_map = dict(zip(plxfpred_df["id"].astype(str), plxfpred_df["classification_prediction"].astype(float)))
    out["PLXFPred"] = out["id"].astype(str).map(plxfpred_map).astype(np.float32)

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
