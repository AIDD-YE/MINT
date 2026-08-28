from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ArchSpec:
    kind: str
    model_dim: int
    protein_layers: int
    ligand_layers: int
    heads: int
    hidden: int
    head_input: int
    chem_dim: int
    morgan_dim: int
    protein_input: int
    atom_input: int

    def describe(self) -> Dict:
        return {
            "kind": self.kind,
            "model_dim": self.model_dim,
            "protein_layers": self.protein_layers,
            "ligand_layers": self.ligand_layers,
            "heads": self.heads,
            "hidden": self.hidden,
            "head_input": self.head_input,
            "chem_dim": self.chem_dim,
            "morgan_dim": self.morgan_dim,
            "protein_input": self.protein_input,
            "atom_input": self.atom_input,
        }


def _leaf_index(keys, prefix):
    layer_keys = []
    for key in keys:
        if key.startswith(prefix):
            remainder = key[len(prefix):]
            first = remainder.split(".")[0]
            if first.isdigit():
                layer_keys.append(int(first))
    return max(layer_keys) if layer_keys else 0


def default_heads(model_dim: int) -> int:
    return max(1, model_dim // 64)


def infer_pocketnet(state: Dict) -> ArchSpec:
    projection = state["prot_enc.proj.weight"]
    model_dim = int(projection.shape[0])
    protein_input = int(projection.shape[1])
    protein_layers = _leaf_index(state.keys(), "prot_enc.enc.layers.") + 1
    ligand_layers = _leaf_index(state.keys(), "lig_enc.convs.") + 1
    atom_input = int(state["lig_enc.x_proj.weight"].shape[1])
    chem_dim = int(state["chem_proj.weight"].shape[1])
    morgan_dim = int(state["morgan_proj.weight"].shape[1])
    head_input = int(state["head.0.weight"].shape[1])
    hidden = int(state["head.0.weight"].shape[0])
    heads = default_heads(model_dim)
    return ArchSpec(
        kind="pocketnet",
        model_dim=model_dim,
        protein_layers=protein_layers,
        ligand_layers=ligand_layers,
        heads=heads,
        hidden=hidden,
        head_input=head_input,
        chem_dim=chem_dim,
        morgan_dim=morgan_dim,
        protein_input=protein_input,
        atom_input=atom_input,
    )


def infer_mlp(state: Dict) -> ArchSpec:
    projection = state["net.0.weight"]
    input_dim = int(projection.shape[1])
    hidden = int(projection.shape[0])
    return ArchSpec(
        kind="mlp",
        model_dim=input_dim,
        protein_layers=0,
        ligand_layers=0,
        heads=0,
        hidden=hidden,
        head_input=input_dim,
        chem_dim=0,
        morgan_dim=0,
        protein_input=input_dim,
        atom_input=0,
    )


def infer_arch(state: Dict, kind: str) -> ArchSpec:
    if kind == "pocketnet":
        return infer_pocketnet(state)
    if kind == "mlp":
        return infer_mlp(state)
    raise ValueError("cannot infer architecture for kind %s" % kind)


def describe_arch(spec: ArchSpec) -> Dict:
    return spec.describe()