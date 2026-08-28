import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

from mint_engine.arch import ArchSpec, infer_arch


class ProteinEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, layers, heads, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, tokens, mask):
        hidden = self.proj(tokens)
        hidden = self.enc(hidden, src_key_padding_mask=~mask)
        return self.ln(hidden)


class LigandGIN(nn.Module):
    def __init__(self, atom_dim, model_dim, layers, dropout):
        super().__init__()
        self.x_proj = nn.Linear(atom_dim, model_dim)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(model_dim, model_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim * 2, model_dim),
            )
            self.convs.append(GINConv(mlp))
        self.lns = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(layers)])
        self.dropout = dropout

    def forward(self, graph):
        hidden = self.x_proj(graph.x)
        for conv, layer_norm in zip(self.convs, self.lns):
            update = conv(hidden, graph.edge_index)
            update = F.gelu(update)
            update = F.dropout(update, p=self.dropout, training=self.training)
            hidden = layer_norm(hidden + update)
        graph_vec = global_mean_pool(hidden, graph.batch)
        return hidden, graph_vec


class CrossAttnBinder(nn.Module):
    def __init__(self, spec: ArchSpec, dropout):
        super().__init__()
        self.spec = spec
        self.prot_enc = ProteinEncoder(
            input_dim=spec.protein_input,
            model_dim=spec.model_dim,
            layers=spec.protein_layers,
            heads=spec.heads,
            dropout=dropout,
        )
        self.lig_enc = LigandGIN(
            atom_dim=spec.atom_input,
            model_dim=spec.model_dim,
            layers=spec.ligand_layers,
            dropout=dropout,
        )
        self.cross = nn.MultiheadAttention(spec.model_dim, spec.heads, dropout=dropout, batch_first=True)
        self.chem_proj = nn.Linear(spec.chem_dim, spec.model_dim)
        self.morgan_proj = nn.Linear(spec.morgan_dim, spec.model_dim)
        self.head = nn.Sequential(
            nn.Linear(spec.head_input, spec.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spec.hidden, 1),
        )

    def forward(self, tokens, mask, graph, chem=None, morgan=None):
        protein_tokens = self.prot_enc(tokens, mask)
        ligand_nodes, ligand_graph = self.lig_enc(graph)
        device = tokens.device
        batch_size = int(graph.batch.max().item()) + 1
        model_dim = ligand_nodes.size(-1)
        counts = torch.bincount(graph.batch, minlength=batch_size).tolist()
        width = max(counts) if counts else 0
        atom_padded = ligand_nodes.new_zeros((batch_size, width, model_dim))
        atom_mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
        cursor = 0
        for i, count in enumerate(counts):
            if count > 0:
                atom_padded[i, :count] = ligand_nodes[cursor:cursor + count]
                atom_mask[i, :count] = True
                cursor += count
        attended, _ = self.cross(protein_tokens, atom_padded, atom_padded, key_padding_mask=~atom_mask, need_weights=False)
        pooled_protein = self._mean(protein_tokens, mask)
        pooled_attended = self._mean(attended, mask)
        features = [pooled_protein, ligand_graph, pooled_attended, pooled_protein * ligand_graph]
        if chem is not None and morgan is not None:
            features.append(self.chem_proj(chem))
            features.append(self.morgan_proj(morgan))
        concat = torch.cat(features, dim=-1)
        return self.head(concat).squeeze(1)

    @staticmethod
    def _mean(tensor, mask, eps=1e-6):
        weights = mask.float().unsqueeze(-1)
        return (tensor * weights).sum(dim=1) / (weights.sum(dim=1) + eps)


class SimpleMLP(nn.Module):
    def __init__(self, spec: ArchSpec, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spec.head_input, spec.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(spec.hidden, 1),
        )

    def forward(self, protein_vec, ligand_vec):
        concat = torch.cat([protein_vec, ligand_vec], dim=1)
        return self.net(concat).squeeze(1)


def build_from_checkpoint(state, kind, dropout=0.1):
    spec = infer_arch(state, kind)
    if kind == "pocketnet":
        model = CrossAttnBinder(spec, dropout)
    else:
        model = SimpleMLP(spec, dropout)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError("state dict mismatch: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    return model, spec

def count_parameters(model):
    return sum(int(parameter.numel()) for parameter in model.parameters())