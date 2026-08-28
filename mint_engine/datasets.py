import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


def clip_score(value, eps=1e-5):
    value = float(value)
    if value != value:
        return 0.5
    return max(eps, min(1.0 - eps, value))


class PairDatasetMemPocketNet(Dataset):
    def __init__(self, frame, protein, ligand, prot_col="prot_key", lig_col="lig_key", y_col="classification_label", teacher_col="classification_prediction_raw", with_teacher=False):
        self.frame = frame.reset_index(drop=True)
        self.protein = protein
        self.ligand = ligand
        self.prot_col = prot_col
        self.lig_col = lig_col
        self.y_col = y_col
        self.teacher_col = teacher_col
        self.with_teacher = bool(with_teacher)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[int(index)]
        prot_key = str(row[self.prot_col])
        lig_key = str(row[self.lig_col])
        prot = self.protein[prot_key]
        lig = self.ligand[lig_key]
        p_tok = torch.from_numpy(prot["token_representation"].astype("float32"))
        x = lig["atom_feature"].float()
        edge_index = lig["edge_index"].long()
        graph = Data(x=x, edge_index=edge_index)
        edge_weight = lig.get("edge_weight", None)
        if edge_weight is not None:
            graph.edge_weight = edge_weight.float()
        chem = lig.get("chemBERTa", None)
        morgan = lig.get("morgan_fingerprint", None)
        chem = chem.float().view(-1) if chem is not None else None
        morgan = morgan.float().view(-1) if morgan is not None else None
        target = torch.tensor(float(row[self.y_col]), dtype=torch.float32)
        if self.with_teacher:
            teacher = torch.tensor(clip_score(row[self.teacher_col]), dtype=torch.float32)
        else:
            teacher = torch.tensor(0.5, dtype=torch.float32)
        return p_tok, graph, chem, morgan, target, teacher


def collate_pairs_pyg(batch):
    p_tok, graphs, chem, morgan, target, teacher = zip(*batch)
    batch_size = len(p_tok)
    feature_dim = p_tok[0].shape[1]
    length_max = max(tensor.shape[0] for tensor in p_tok)
    padded = torch.zeros((batch_size, length_max, feature_dim), dtype=torch.float32)
    padding_mask = torch.zeros((batch_size, length_max), dtype=torch.bool)
    for i, tensor in enumerate(p_tok):
        length = tensor.shape[0]
        padded[i, :length] = tensor
        padding_mask[i, :length] = True
    graphs = Batch.from_data_list(list(graphs))
    chem_stack = torch.stack(list(chem), 0) if all(value is not None for value in chem) else None
    morgan_stack = torch.stack(list(morgan), 0) if all(value is not None for value in morgan) else None
    target = torch.stack(list(target), 0)
    teacher = torch.stack(list(teacher), 0)
    return padded, padding_mask, graphs, chem_stack, morgan_stack, target, teacher


class PairDatasetMemMLP(Dataset):
    def __init__(self, frame, protein, ligand, prot_col="prot_key", lig_col="lig_key", y_col="classification_label", teacher_col="classification_prediction_raw", with_teacher=False, vector_policy="prefer_chem"):
        self.frame = frame.reset_index(drop=True)
        self.protein = protein
        self.ligand = ligand
        self.prot_col = prot_col
        self.lig_col = lig_col
        self.y_col = y_col
        self.teacher_col = teacher_col
        self.with_teacher = bool(with_teacher)
        self.vector_policy = vector_policy

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[int(index)]
        prot_key = str(row[self.prot_col])
        lig_key = str(row[self.lig_col])
        prot = self.protein[prot_key]
        lig = self.ligand[lig_key]
        tokens = prot["token_representation"].astype("float32")
        protein_vec = torch.from_numpy(tokens).mean(dim=0)
        chem = lig.get("chemBERTa", None)
        morgan = lig.get("morgan_fingerprint", None)
        if self.vector_policy == "prefer_chem":
            if chem is not None:
                ligand_vec = chem.float().view(-1)
            elif morgan is not None:
                ligand_vec = morgan.float().view(-1)
            else:
                raise ValueError("ligand %s has neither chemBERTa nor morgan_fingerprint" % lig_key)
        elif self.vector_policy == "prefer_morgan":
            if morgan is not None:
                ligand_vec = morgan.float().view(-1)
            elif chem is not None:
                ligand_vec = chem.float().view(-1)
            else:
                raise ValueError("ligand %s has neither representation" % lig_key)
        else:
            raise ValueError("unknown vector_policy %s" % self.vector_policy)
        target = torch.tensor(float(row[self.y_col]), dtype=torch.float32)
        if self.with_teacher:
            teacher = torch.tensor(clip_score(row[self.teacher_col]), dtype=torch.float32)
        else:
            teacher = torch.tensor(0.5, dtype=torch.float32)
        return protein_vec, ligand_vec, target, teacher


def collate_mlp(batch):
    protein_vec, ligand_vec, target, teacher = zip(*batch)
    return torch.stack(protein_vec, 0), torch.stack(ligand_vec, 0), torch.stack(target, 0), torch.stack(teacher, 0)


def build_dataset(kind, frame, protein, ligand):
    if kind == "pocketnet":
        return PairDatasetMemPocketNet(frame, protein, ligand), collate_pairs_pyg
    return PairDatasetMemMLP(frame, protein, ligand), collate_mlp