#!/usr/bin/env python3
"""Prepare, then train one facing-aware native CzechLynx MiewID trial (arm 'classes' or 'subcenter').

The loop is the v2 FP32 loop (utils/finetune_czechlynx_miewid.py), copied unchanged except for the
class definition, the optional sub-center head and the same-facing checkpoint selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_embeddings import load_miewid
from reid_demo.benchmark_miewid_adaptation import (
    EncounterBatchSampler, ImageDataset, build_transform, extract_matrix, make_head, native_state_digest,
    training_mode,
)
from reid_demo import benchmark_facing_adaptation as fm
from reid_demo.benchmark_input_audit import load_inputs, validate_review_receipt, verify_prepared
from utils.finetune_czechlynx_miewid import gradient_diagnostics, seed_worker


def run_training(args):
    settings = dict(fm.DEFAULT_SETTINGS)
    for key in ("epochs", "patience", "identities_per_batch", "images_per_identity", "seed"):
        settings[key] = getattr(args, key)
    settings["arm"] = args.arm
    out = Path(args.out_dir).resolve()
    if not args.prepare_only and not (out / "inner_plan.json").is_file():
        raise ValueError("Run --prepare-only and inspect the inner plan before training.")
    expected = None if args.allow_new_inner_plan else fm.V2_INNER_PLAN_FINGERPRINT
    frame, _, manifest = fm.prepare_trial(args.metadata, args.protocol_plan, args.audit_receipt, args.labels_dir,
                                          out, settings, expected_inner=expected)
    roles = manifest["inner_plan"]["roles"]
    labels_table, _ = fm.load_label_table(args.labels_dir, [i for r in roles.values() for i in r["image_ids"]])
    facing_of = dict(zip(labels_table.image_id, labels_table.facing.astype(int)))
    print(f"Arm {settings['arm']}; inner plan {manifest['inner_plan']['fingerprint'][:16]}; train "
          f"{roles['train']['inventory']}; validation {roles['validation']['inventory']}", flush=True)
    steps = max(1, int(np.ceil(roles["train"]["inventory"]["images"] /
                              (settings["identities_per_batch"] * settings["images_per_identity"]))))
    print(f"Maximum {settings['epochs']} epochs x {steps} balanced batches; {settings['warmup_epochs']} head warmup epoch.", flush=True)
    if args.prepare_only:
        print(f"Prepared without neural imports/GPU allocation: {out / 'inner_plan.json'}", flush=True)
        return manifest
    started = time.perf_counter()
    manifest["status"] = "running"
    manifest["runtime"] = {"workers": args.workers, "threads": args.threads, "chunk_size": args.chunk_size}
    write_json(out / "manifest.json", manifest)
    context = {"stage": "initialization", "epoch": None, "batch": None, "image_ids": [],
               "optimizer_step_applied": False}
    named_parameters = []
    try:
        import torch
        from torch.utils.data import DataLoader
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(args.threads)
        torch.manual_seed(settings["seed"])
        np.random.seed(settings["seed"])
        random.seed(settings["seed"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manifest["runtime"].update(device=str(device),
                                   training_precision={"parameter_dtype": "float32", "input_dtype": "float32",
                                                       "autocast": False, "gradient_scaling": False},
                                   evaluation_precision={"parameter_dtype": "float32", "autocast": device.type == "cuda",
                                                         "autocast_dtype": "float16" if device.type == "cuda" else None,
                                                         "normalization_dtype": "float32", "matrix_dtype": "float32"},
                                   cuda_version=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
        indexed = frame.set_index("image_id", drop=False)
        train = indexed.loc[roles["train"]["image_ids"]].reset_index(drop=True)
        val = indexed.loc[roles["validation"]["image_ids"]].reset_index(drop=True)
        val_facing = np.asarray([facing_of[i] for i in val.image_id], int)
        # The sampler and dataset group by the 'identity' column, so the training class key replaces it there only.
        train_classes = train.assign(identity=fm.class_keys(train, [facing_of[i] for i in train.image_id], settings["arm"]))
        model, model_info, size = load_miewid(device, revision=settings["revision"])
        model.float()
        if size != settings["image_size"] or model_info["embedding_dim"] != settings["embedding_dim"]:
            raise ValueError("Publisher native architecture/dimensions differ.")
        manifest["model_info"] = model_info
        manifest["initial_native_state_sha256"] = native_state_digest(model.state_dict())
        publisher_final = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
                           if name.startswith("final.")}
        embedding = [(name, p) for name, p in model.named_parameters() if not name.startswith("final.")]
        if not embedding:
            raise ValueError("No native embedding parameters found.")
        manifest["embedding_parameter_names"] = [name for name, _ in embedding]
        labels = {key: i for i, key in enumerate(sorted(train_classes.identity.unique()))}
        manifest["training_class_map"] = labels
        manifest["training_class_counts"] = {"classes": len(labels), "animals": int(train.identity.nunique())}
        if settings["arm"] == "subcenter":
            head = fm.make_subcenter_head(settings["embedding_dim"], len(labels), k=settings["subcenters"],
                                          scale=settings["arcface_scale"], margin=settings["arcface_margin"])
        else:
            head = make_head(settings["embedding_dim"], len(labels), scale=settings["arcface_scale"],
                             margin=settings["arcface_margin"])
        head = head.to(device=device, dtype=torch.float32)
        named_parameters = [("model." + name, p) for name, p in embedding]
        named_parameters.extend(("head." + name, p) for name, p in head.named_parameters())
        optimizer = torch.optim.AdamW([
            {"params": [p for _, p in embedding], "lr": settings["embedding_lr"]},
            {"params": list(head.parameters()), "lr": settings["head_lr"]}], weight_decay=settings["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"] - settings["warmup_epochs"])
        sampler = EncounterBatchSampler(train_classes, settings["identities_per_batch"], settings["images_per_identity"],
                                        settings["seed"])
        generator = torch.Generator().manual_seed(settings["seed"])
        loader = DataLoader(ImageDataset(train_classes, build_transform(settings, training=True), labels),
                            batch_sampler=sampler, num_workers=args.workers, worker_init_fn=seed_worker,
                            generator=generator, pin_memory=device.type == "cuda")
        from reid_demo.benchmark_diagnostics import measure_appearance
        appearance = {row["image_id"]: measure_appearance(row) for row in val.to_dict("records")}
        write_json(out / "holdout_appearance.json", appearance)
        epoch_logs = []

        def evaluate(epoch):
            context.update(stage="inner_validation", epoch=epoch, batch=None, image_ids=[], optimizer_step_applied=False)
            matrix = extract_matrix(model, val, settings, device, batch_size=sampler.p * sampler.k, workers=args.workers)
            metrics, rows = fm.epoch_metrics(val, matrix, val_facing, appearance=appearance, chunk_size=args.chunk_size)
            if metrics["macro_same_facing_top1"] is None or metrics["macro_recall_at_50"] is None:
                raise ValueError("Inner holdout has no eligible same-facing/cross-encounter queries.")
            write_json(out / "inner_validation_latest.json", {"epoch": epoch, "metrics": metrics, "queries": rows})
            return metrics
        with threadpool_limits(limits=args.threads):
            timer = time.perf_counter()
            baseline = evaluate(0)
            best, best_epoch = baseline, 0
            manifest["baseline_metrics"] = baseline
            fm.save_checkpoint(out / "best.pt", model, head, manifest, 0, baseline, model_info)
            epoch_logs.append({"epoch": 0, "metrics": baseline, "seconds": time.perf_counter() - timer})
            print(f"Frozen inner holdout: same-facing top1={baseline['macro_same_facing_top1']:.4f}; "
                  f"recall@50={baseline['macro_recall_at_50']:.4f}; top1={baseline['macro_top1']:.4f}", flush=True)
            stale, gradient_verified = 0, False
            for epoch in range(1, settings["epochs"] + 1):
                timer = time.perf_counter()
                warmup = epoch <= settings["warmup_epochs"]
                training_mode(model, warmup=warmup)
                head.train()
                sampler.set_epoch(epoch)
                losses, observations = 0., 0
                for batch, (images, targets, batch_ids) in enumerate(loader, 1):
                    context.update(stage="training_forward", epoch=epoch, batch=batch,
                                   image_ids=[str(value) for value in batch_ids], optimizer_step_applied=False)
                    images, targets = images.to(device=device, dtype=torch.float32), targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=False):
                        features = model(images)
                        context["stage"] = "training_embedding_check"
                        if (features.ndim != 2 or features.shape != (len(images), settings["embedding_dim"])
                                or not torch.isfinite(features).all() or (features.float().norm(dim=1) == 0).any()):
                            raise ValueError("Invalid training embedding shape/values.")
                        context["stage"] = "training_loss"
                        loss = torch.nn.functional.cross_entropy(head(features.float(), targets), targets)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training loss.")
                        context["stage"] = "training_backward"
                        loss.backward()
                        context["stage"] = "gradient_clipping"
                        parameters = [p for _, p in named_parameters if p.grad is not None]
                        norm = torch.nn.utils.clip_grad_norm_(parameters, settings["clip_norm"], error_if_nonfinite=True)
                        context["stage"] = "embedding_gradient_proof"
                        proof = None
                        if not warmup and not gradient_verified:
                            effective = [(name, p) for name, p in embedding if p.grad is not None and p.grad.abs().max() > 0]
                            if not effective:
                                raise ValueError("No effective gradient in the intended native embedding path.")
                            name, param = max(effective, key=lambda item: float(item[1].grad.abs().max()))
                            proof = (name, param, param.detach().clone())
                        context["stage"] = "optimizer_step"
                        optimizer.step()
                        context["optimizer_step_applied"] = True
                    if proof is not None:
                        context["stage"] = "embedding_update_proof"
                        name, param, before = proof
                        if torch.equal(param.detach(), before):
                            raise ValueError("Embedding gradients did not produce an effective parameter update.")
                        manifest["gradient_proof"] = {"epoch": epoch, "batch": batch, "updated_parameter": name,
                                                      "finite_global_gradient_norm": float(norm)}
                        gradient_verified = True
                    losses += float(loss.detach()) * len(images)
                    observations += len(images)
                    if batch == 1 or batch % 200 == 0 or batch == len(loader):
                        elapsed = time.perf_counter() - timer
                        print(f"Epoch {epoch}/{settings['epochs']} batch {batch}/{len(loader)}; loss {float(loss.detach()):.4f}; "
                              f"remaining train portion ~{elapsed / batch * (len(loader) - batch):.0f}s", flush=True)
                context.update(stage="model_state_check", batch=None, image_ids=[], optimizer_step_applied=False)
                for name, tensor in model.state_dict().items():
                    if not torch.isfinite(tensor).all():
                        context["parameter"] = name
                        raise ValueError(f"Nonfinite model state: {name}")
                for name, value in publisher_final.items():
                    if not torch.equal(model.state_dict()[name].cpu(), value):
                        raise ValueError("Publisher unused final layer changed.")
                if warmup and native_state_digest(model.state_dict()) != manifest["initial_native_state_sha256"]:
                    raise ValueError("Native embedding state changed during head-only warmup.")
                metrics = evaluate(epoch)
                improved = not warmup and fm.selection_key(metrics) > fm.selection_key(best)
                if improved:
                    best, best_epoch, stale = metrics, epoch, 0
                    fm.save_checkpoint(out / "best.pt", model, head, manifest, epoch, metrics, model_info)
                elif not warmup:
                    stale += 1
                fm.save_checkpoint(out / "last.pt", model, head, manifest, epoch, metrics, model_info)
                duration = time.perf_counter() - timer
                epoch_logs.append({"epoch": epoch, "warmup": warmup, "mean_loss": losses / observations,
                                   "observations": observations, "sampler": dict(sampler.audit), "metrics": metrics,
                                   "seconds": duration, "learning_rates": [g["lr"] for g in optimizer.param_groups]})
                write_json(out / "epochs.json", epoch_logs)
                print(f"Epoch {epoch}: {duration:.1f}s; same-facing top1={metrics['macro_same_facing_top1']:.4f}; "
                      f"recall@50={metrics['macro_recall_at_50']:.4f}; top1={metrics['macro_top1']:.4f}; "
                      f"best epoch={best_epoch}", flush=True)
                if not warmup:
                    scheduler.step()
                if not warmup and stale >= settings["patience"]:
                    break
            if not gradient_verified:
                raise ValueError("Trial did not verify an embedding update after warmup.")
        _, _, current_binding = load_inputs(args.metadata, args.protocol_plan)
        if current_binding != manifest["binding"] or fm.implementation() != manifest["implementation"]:
            raise ValueError("Training source/protocol/implementation changed during the trial.")
        if file_digest(manifest["facing_labels"]["path"]) != manifest["facing_labels"]["sha256"]:
            raise ValueError("Facing labels changed during the trial.")
        verify_prepared(train)
        verify_prepared(val)
        validate_review_receipt(args.audit_receipt, frame, manifest["binding"])
        manifest.update(status="complete", best_epoch=best_epoch, best_metrics=best,
                        inner_holdout_improved=fm.selection_key(best) > fm.selection_key(baseline),
                        elapsed_seconds=time.perf_counter() - started)
        manifest["artifacts"] = {name: file_digest(out / name) for name in
            ("inner_plan.json", "holdout_appearance.json", "inner_validation_latest.json", "epochs.json", "best.pt", "last.pt")}
        write_json(out / "manifest.json", manifest)
        print("Frozen epoch zero remains best." if best_epoch == 0 else f"Selected checkpoint: {out / 'best.pt'} (epoch {best_epoch}).",
              flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        manifest["failure_context"] = dict(context)
        if context["stage"] in {"training_forward", "training_embedding_check", "training_loss", "training_backward",
                                "gradient_clipping", "model_state_check"}:
            details = {**context, "kind": context["stage"]}
            if context["stage"] in {"training_backward", "gradient_clipping"}:
                try:
                    details.update(gradient_diagnostics(named_parameters))
                except Exception as diagnostic_error:
                    details["diagnostic_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            manifest["numerical_failure"] = details
        try:
            write_json(out / "manifest.json", manifest)
        except Exception as persistence_error:
            print(f"Could not save failed manifest: {type(persistence_error).__name__}: {persistence_error}", file=sys.stderr)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata", "protocol-plan", "audit-receipt", "labels-dir", "out-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", required=True, choices=fm.ARMS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-new-inner-plan", action="store_true",
                        help="Tests only: do not require the v2 inner-plan fingerprint.")
    for name, default in (("epochs", 10), ("patience", 3), ("identities-per-batch", 4),
                          ("images-per-identity", 2), ("seed", 42), ("threads", 4), ("workers", 4), ("chunk-size", 64)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.workers < 0 or args.chunk_size < 1:
        parser.error("Invalid thread/worker/chunk settings.")
    try:
        run_training(args)
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
