import os
import time

import numpy as np
from torch.utils.data import DataLoader

from mint_engine.arch import infer_arch
from mint_engine.cli import RunOptions, options_from_args
from mint_engine.datasets import build_dataset
from mint_engine.environment import probe_environment
from mint_engine.features import FeatureCacheLoader
from mint_engine.inference import dispatch_predict, sanitize_errors
from mint_engine.io_utils import ensure_dir, path_exists, stream_write_csv
from mint_engine.manifest import ManifestReader
from mint_engine.metrics import metric_block, verify_metrics
from mint_engine.models import build_from_checkpoint, count_parameters
from mint_engine.reporting import FoldResult, collect_integrity, compose_summary, dump_integrity, dump_summary, print_table


class Pipeline:
    def __init__(self, options: RunOptions):
        self.options = options
        self.environment = probe_environment()
        self.device = self.environment.choose_device(options.device)
        self.manifest = None
        self.features = None
        self.arch = None

    def prepare(self):
        options = self.options
        ensure_dir(options.out_dir)
        manifest_reader = ManifestReader(options.manifest).load()
        self.manifest = manifest_reader
        prot_keys, lig_keys = manifest_reader.keys()
        loader = FeatureCacheLoader(options.prot_dir, options.lig_dir, verify=True)
        loader.preload(prot_keys, lig_keys)
        self.features = loader
        frame = manifest_reader.frame
        dataset, collate = build_dataset(options.kind, frame, loader.protein_cache(), loader.ligand_cache())
        self.loader = DataLoader(
            dataset,
            batch_size=options.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=options.workers,
        )
        return self

    def infer(self):
        options = self.options
        lines = self.environment.render("[%s]" % options.exp)
        for line in lines:
            print(line)
        manifest_stats = self.manifest.collect_stats()
        print("[%s] manifest rows=%d pos=%d neg=%d prot=%d lig=%d" % (
            options.exp,
            manifest_stats["rows"],
            manifest_stats["positive"],
            manifest_stats["negative"],
            manifest_stats["unique_proteins"],
            manifest_stats["unique_ligands"],
        ))
        feature_stats = self.features.stats.to_dict()
        print("[%s] features cached prot=%d lig=%d" % (
            options.exp,
            feature_stats["protein_cached"],
            feature_stats["ligand_cached"],
        ))

        probe_ckpt = os.path.join(options.ckpt_dir, "fold%d" % options.folds[0], "stage2_best.pt")
        state = self._load_checkpoint(probe_ckpt)
        if self.arch is None:
            model, spec = build_from_checkpoint(state, options.kind)
            self.arch = spec.describe()
            self.model = model
            self.arch["parameters"] = count_parameters(model)
        print("[%s] architecture model_dim=%d params=%s" % (
            options.exp,
            self.arch.get("model_dim"),
            self.arch.get("parameters"),
        ))

        rows = []
        results = []
        aucs = []
        for fold in options.folds:
            chk = os.path.join(options.ckpt_dir, "fold%d" % fold, "stage2_best.pt")
            if not path_exists(chk):
                print("[%s] fold%d checkpoint missing, skipping" % (options.exp, fold))
                continue
            started = time.time()
            result = self._run_fold(fold, chk)
            result.seconds = time.time() - started
            results.append(result)
            aucs.append(result.metrics["auc"])
            rows.append({
                "fold": result.fold,
                "n": result.samples,
                "auc": "%.4f" % result.metrics["auc"],
                "manual_auc": "%.4f" % result.metrics["manual_auc"],
                "aupr": "%.4f" % result.metrics["aupr"],
                "acc": "%.4f" % result.metrics["accuracy"],
            })
            print("[%s] fold%d auc=%.4f manual_auc=%.4f aupr=%.4f acc=%.4f" % (
                options.exp,
                result.fold,
                result.metrics["auc"],
                result.metrics["manual_auc"],
                result.metrics["aupr"],
                result.metrics["accuracy"],
            ))
        print_table(rows, ["fold", "n", "auc", "manual_auc", "aupr", "acc"])
        return results

    def _run_fold(self, fold, checkpoint):
        options = self.options
        frame = self.manifest.frame
        state = self._load_checkpoint(checkpoint)
        model, spec = build_from_checkpoint(state, options.kind)
        model = model.to(self.device).eval()
        labels, probabilities = dispatch_predict(options.kind, model, self.loader, self.device)
        probabilities = sanitize_errors(labels, probabilities)
        metrics = metric_block(labels, probabilities)
        agreement = verify_metrics(metrics)
        result = FoldResult(
            fold=fold,
            checkpoint=checkpoint,
            samples=int(len(labels)),
            metrics=metrics,
            agreement=agreement,
            labels=np.asarray(labels),
            probabilities=np.asarray(probabilities),
        )
        if options.write_integrity:
            csv_path = os.path.join(options.out_dir, "%s_fold%d_preds.csv" % (options.exp, fold))
            stream_write_csv(csv_path, ["classification_label", "student_prediction_prob"],
                             zip(result.labels.tolist(), result.probabilities.tolist()))
        return result

    def finish(self, results):
        options = self.options
        if not results:
            return
        summary = compose_summary(
            options.exp,
            options.kind,
            self.device,
            results,
            self.arch,
            self.features.stats.to_dict(),
            self.manifest.stats,
            self.environment.describe(),
        )
        dump_summary(os.path.join(options.out_dir, "%s_summary.json" % options.exp), summary)
        if options.write_integrity:
            integrity = collect_integrity(options.ckpt_dir, options.prot_dir, options.lig_dir, options.folds)
            dump_integrity(os.path.join(options.out_dir, "%s_integrity.json" % options.exp), integrity)
        aucs = [result.metrics["auc"] for result in results if result.metrics.get("auc") == result.metrics.get("auc")]
        if aucs:
            mean_auc = float(np.mean(aucs))
            print("[%s] MEAN auc=%.4f" % (options.exp, mean_auc))

    def _load_checkpoint(self, path):
        import torch
        return torch.load(path, map_location=self.device)


def run_pipeline(args) -> int:
    options = options_from_args(args)
    pipeline = Pipeline(options).prepare()
    results = pipeline.infer()
    pipeline.finish(results)
    return 0