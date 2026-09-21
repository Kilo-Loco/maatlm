"""Train a SystemOneModel on JSONL decisions with a proper scoring rule.

    python -m maatlm.train \
        --base Qwen/Qwen3-1.7B-Base --train data/train.jsonl --val data/val.jsonl \
        --out runs/jev-open-1.7b --epochs 2 --batch 8 --grad-accum 4 --lr 1e-5 --bf16

Options:
  --lora R          train a LoRA adapter of rank R instead of all weights (peft)
  --rule log|brier  proper scoring rule
  --max-state-tokens N   TRAINING-ONLY truncation (memory); never saved into the checkpoint
  --serve-max-state-tokens N  what the saved checkpoint reports as its state budget (default 32768)
  --rps W           add W * ranked-probability-score for `score` questions (ordinal-aware)
  --consistency W   add W * JS(state, paraphrase) for examples that carry `paraphrases`
  --shuffle-options randomise option order per sample (see data.py)
  --tiny            use the offline random tiny backbone (smoke tests only)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .data import SystemOneDataset, collate_train, read_jsonl
from .losses import batch_loss
from .model import SystemOneModel


def load_model(args) -> SystemOneModel:
    if args.tiny:
        from .tiny import tiny_model
        return tiny_model(attn_implementation=args.attn)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    m = SystemOneModel.from_pretrained(
        args.base, torch_dtype=dtype, attn_implementation=args.attn, max_state_tokens=args.max_state_tokens
    )
    return m


def maybe_lora(m: SystemOneModel, r: int, alpha: Optional[int] = None):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha or 2 * r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    m.backbone = get_peft_model(m.backbone, cfg)
    m.backbone.print_trainable_parameters()
    return m


@torch.no_grad()
def evaluate(m: SystemOneModel, loader, device, rule: str, rps: float = 0.0, consistency: float = 0.0):
    m.eval()
    tot, n, per = 0.0, 0, {"choice": [], "score": [], "noul": [], "consistency": []}
    for batch, targets in loader:
        pairs = getattr(batch, "pairs", None)
        batch = batch.to(device)
        raw = m(batch)
        loss, pt = batch_loss(raw, targets, batch.layouts, rule, rps_weight=rps, pairs=pairs, consistency_weight=consistency)
        tot += loss.item()
        n += 1
        for k, v in pt.items():
            if v is not None:
                per[k].append(v)
    m.train()
    return tot / max(n, 1), {k: (sum(v) / len(v) if v else None) for k, v in per.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rule", choices=["log", "brier"], default="log")
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--rps", type=float, default=0.5, help="weight of the ranked probability score term for score questions")
    ap.add_argument("--consistency", type=float, default=0.1, help="weight of the paraphrase-consistency (JS) term; 0 disables")
    ap.add_argument("--shuffle-options", action="store_true")
    ap.add_argument("--lora", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--max-state-tokens", type=int, default=32768,
                    help="training-only truncation for memory; NOT persisted to the checkpoint")
    ap.add_argument("--serve-max-state-tokens", type=int, default=32768,
                    help="state budget written into the saved checkpoint (serving limit)")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = load_model(args)
    if args.lora:
        m = maybe_lora(m, args.lora)
    if args.grad_checkpoint:
        m.backbone.gradient_checkpointing_enable()
        if hasattr(m.backbone, "enable_input_require_grads"):
            m.backbone.enable_input_require_grads()
    m.to(device)
    # temperatures are fitted post hoc; keep them fixed at 1 during training
    for p in m.log_temp.values():
        p.requires_grad_(False)

    train_ex = read_jsonl(args.train)
    use_para = args.consistency > 0 and any(e.paraphrases for e in train_ex)
    train_ds = SystemOneDataset(
        m, train_ex, shuffle_options=args.shuffle_options, smoothing=args.smoothing, seed=args.seed, paraphrases=use_para
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_train(m), num_workers=args.workers)
    val_dl = None
    if args.val:
        val_ds = SystemOneDataset(m, read_jsonl(args.val), paraphrases=use_para)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_train(m), num_workers=args.workers)

    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warm = int(total_steps * args.warmup)

    def lr_at(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        prog = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    m.train()
    step, micro, t0, best = 0, 0, time.time(), float("inf")
    done = False
    while not done:
        for batch, targets in train_dl:
            pairs = getattr(batch, "pairs", None)
            batch = batch.to(device)
            raw = m(batch)
            loss, per_type = batch_loss(
                raw, targets, batch.layouts, args.rule, rps_weight=args.rps, pairs=pairs, consistency_weight=args.consistency
            )
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0], "per_type": per_type, "t": time.time() - t0}
            if step % 10 == 0 or step == 1:
                print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            if val_dl is not None and (step % args.eval_every == 0 or step == total_steps):
                vl, vpt = evaluate(m, val_dl, device, args.rule, args.rps, args.consistency)
                print(json.dumps({"step": step, "val_loss": vl, "val_per_type": vpt}), flush=True)
                log.write(json.dumps({"step": step, "val_loss": vl, "val_per_type": vpt}) + "\n")
                if vl < best:
                    best = vl
                    _save(m, os.path.join(args.out, "best"), args)
            if args.save_every and step % args.save_every == 0:
                _save(m, os.path.join(args.out, f"step{step}"), args)
            if step >= total_steps:
                done = True
                break
    _save(m, os.path.join(args.out, "final"), args, final=True)
    log.close()
    print("done", {"steps": step, "best_val": best})


def _save(m: SystemOneModel, path: str, args, final: bool = False):
    # The training truncation is a memory decision, not a capability one. Persisting it
    # would silently cut long states at eval/serve time (it did, once: 33% of JevBench's
    # hard tier is over 1024 tokens). Always save the serving budget instead.
    m.max_state_tokens = args.serve_max_state_tokens
    if args.lora and not final:
        # mid-run: save the adapter only (merging would end training)
        os.makedirs(path, exist_ok=True)
        m.backbone.save_pretrained(path)
        with open(os.path.join(path, "BASE"), "w") as f:
            f.write(args.base)
    elif args.lora and final:
        # merge the adapter so the final checkpoint is a plain HF model
        m.backbone = m.backbone.merge_and_unload()
        m.save(path)
    else:
        m.save(path)


if __name__ == "__main__":
    main()
