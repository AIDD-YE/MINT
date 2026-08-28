import os
from typing import Any, Dict, List, Optional, Sequence

from mint_engine.io_utils import path_exists, read_pickle


class FeatureStats:
    def __init__(self):
        self.protein_seen = 0
        self.ligand_seen = 0
        self.protein_lengths = []
        self.protein_dims = set()
        self.ligand_atom_counts = []
        self.ligand_dims = set()
        self.ligand_edge_counts = []
        self.ligand_with_chem = 0
        self.ligand_with_morgan = 0
        self.ligand_with_weights = 0
        self.missing = []
        self.total_loaded_bytes = 0

    def note_protein(self, token):
        self.protein_seen += 1
        self.protein_lengths.append(int(token.shape[0]))
        self.protein_dims.add(int(token.shape[1]))

    def note_ligand(self, obj, path):
        self.ligand_seen += 1
        atoms = obj["atom_feature"]
        self.ligand_atom_counts.append(int(atoms.shape[0]))
        self.ligand_dims.add(int(atoms.shape[1]))
        edge_index = obj["edge_index"]
        self.ligand_edge_counts.append(int(edge_index.shape[1]))
        if obj.get("chemBERTa") is not None:
            self.ligand_with_chem += 1
        if obj.get("morgan_fingerprint") is not None:
            self.ligand_with_morgan += 1
        if obj.get("edge_weight") is not None:
            self.ligand_with_weights += 1
        try:
            self.total_loaded_bytes += os.path.getsize(path)
        except OSError:
            pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protein_cached": self.protein_seen,
            "ligand_cached": self.ligand_seen,
            "protein_length_min": min(self.protein_lengths) if self.protein_lengths else None,
            "protein_length_max": max(self.protein_lengths) if self.protein_lengths else None,
            "protein_length_mean": round(float(sum(self.protein_lengths) / len(self.protein_lengths)), 2) if self.protein_lengths else None,
            "protein_dims_observed": sorted(self.protein_dims),
            "ligand_atom_min": min(self.ligand_atom_counts) if self.ligand_atom_counts else None,
            "ligand_atom_max": max(self.ligand_atom_counts) if self.ligand_atom_counts else None,
            "ligand_atom_mean": round(float(sum(self.ligand_atom_counts) / len(self.ligand_atom_counts)), 2) if self.ligand_atom_counts else None,
            "ligand_dims_observed": sorted(self.ligand_dims),
            "ligand_edge_count_total": sum(self.ligand_edge_counts),
            "ligand_with_chem": self.ligand_with_chem,
            "ligand_with_morgan": self.ligand_with_morgan,
            "ligand_with_weights": self.ligand_with_weights,
            "missing_keys": self.missing,
            "loaded_bytes": self.total_loaded_bytes,
        }


class FeatureCacheLoader:
    def __init__(self, prot_dir: str, lig_dir: str, verify: bool = True):
        self.prot_dir = prot_dir
        self.lig_dir = lig_dir
        self.verify = verify
        self._protein: Dict[str, Any] = {}
        self._ligand: Dict[str, Any] = {}
        self.stats = FeatureStats()

    def _resolve_protein_path(self, key: str) -> str:
        path = os.path.join(self.prot_dir, "%s.pkl" % key)
        if not path_exists(path):
            self.stats.missing.append("prot:%s" % key)
            raise FileNotFoundError("protein feature cache missing for key %s" % key)
        return path

    def _resolve_ligand_path(self, key: str) -> str:
        path = os.path.join(self.lig_dir, "%s.pkl" % key)
        if not path_exists(path):
            self.stats.missing.append("lig:%s" % key)
            raise FileNotFoundError("ligand feature cache missing for key %s" % key)
        return path

    def protein(self, key: str) -> Any:
        if key in self._protein:
            return self._protein[key]
        path = self._resolve_protein_path(key)
        obj = read_pickle(path)
        if "token_representation" not in obj:
            raise ValueError("protein cache %s lacks token_representation" % key)
        if self.verify:
            self.stats.note_protein(obj["token_representation"])
        self._protein[key] = obj
        return obj

    def ligand(self, key: str) -> Any:
        if key in self._ligand:
            return self._ligand[key]
        path = self._resolve_ligand_path(key)
        obj = read_pickle(path)
        for required in ("atom_feature", "edge_index"):
            if required not in obj:
                raise ValueError("ligand cache %s lacks %s" % (key, required))
        if self.verify:
            self.stats.note_ligand(obj, path)
        self._ligand[key] = obj
        return obj

    def preload(self, prot_keys: Sequence[str], lig_keys: Sequence[str]) -> None:
        for key in prot_keys:
            self.protein(str(key))
        for key in lig_keys:
            self.ligand(str(key))

    def protein_cache(self) -> Dict[str, Any]:
        return self._protein

    def ligand_cache(self) -> Dict[str, Any]:
        return self._ligand