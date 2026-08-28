from collections import Counter

import pandas as pd


REQUIRED_COLUMNS = ("prot_key", "lig_key", "classification_label")


class ManifestReader:
    def __init__(self, path: str):
        self.path = path
        self.frame = None
        self.stats = {}

    def load(self):
        frame = pd.read_csv(self.path)
        for column in REQUIRED_COLUMNS:
            if column not in frame.columns:
                raise ValueError("manifest missing required column: %s" % column)
        frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
        frame["classification_label"] = frame["classification_label"].astype(int)
        labels = set(frame["classification_label"].unique())
        if not labels.issubset({0, 1}):
            raise ValueError("classification_label must take values in {0, 1}")
        self.frame = frame
        return self

    def collect_stats(self):
        frame = self.frame
        stats = {
            "rows": int(len(frame)),
            "positive": int((frame["classification_label"] == 1).sum()),
            "negative": int((frame["classification_label"] == 0).sum()),
            "unique_proteins": int(frame["prot_key"].astype(str).nunique()),
            "unique_ligands": int(frame["lig_key"].astype(str).nunique()),
        }
        duplicated_pairs = frame.duplicated(subset=["prot_key", "lig_key"]).sum()
        stats["duplicated_pairs"] = int(duplicated_pairs)
        ligand_rows = frame["lig_key"].astype(str).value_counts()
        protein_rows = frame["prot_key"].astype(str).value_counts()
        stats["max_ligand_reuse"] = int(ligand_rows.max())
        stats["max_protein_reuse"] = int(protein_rows.max())
        stats["ligand_reuse_distribution"] = sorted(Counter(ligand_rows.tolist()).items())
        stats["protein_reuse_distribution"] = sorted(Counter(protein_rows.tolist()).items())
        self.stats = stats
        return stats

    def keys(self):
        frame = self.frame
        prot_keys = sorted(frame["prot_key"].astype(str).unique().tolist())
        lig_keys = sorted(frame["lig_key"].astype(str).unique().tolist())
        return prot_keys, lig_keys

    def frame(self):
        return self.frame