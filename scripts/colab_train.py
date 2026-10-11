"""Run dev search or final training. Resumable: finished runs (summary.json) are skipped and
interrupted runs resume from last.pt (config-hash checked).

  python scripts/colab_train.py dev   --work /content/work --max-epochs 12
  python scripts/colab_train.py final --work /content/work --max-epochs 12 --lr cnn=1e-3 resnet18=3e-4
Source data only; external data is never read here.
"""
import argparse, json, sys, time
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from plantrel import trainer

GRID = {"cnn": [5e-4, 1e-3, 3e-3], "resnet18": [1e-4, 3e-4, 1e-3]}


def cfg(a, model, lr, seed, tag):
    C = Path(a.work) / "cache"
    return dict(model=model, seed=seed, lr=lr, weight_decay=a.weight_decay, batch_size=a.batch_size,
                max_epochs=a.max_epochs, patience=a.patience, warmup_epochs=1, cache=str(C / "source.npy"),
                labels=str(C / "source_labels.npy"), splits=str(C / "source_splits.npy"),
                run_dir=str(Path(a.work) / "runs" / tag), num_classes=23, amp=True, pretrained=True,
                max_train_batches=a.max_train_batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "final"]); ap.add_argument("--work", default="/content/work")
    ap.add_argument("--max-epochs", type=int, default=12); ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=["cnn", "resnet18"]); ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--lr", nargs="*", default=[], help="model=lr pairs for the final stage")
    a = ap.parse_args()
    jobs = []
    if a.stage == "dev":
        jobs = [(m, lr, 1, f"dev_{m}_lr{lr:g}_s1") for m in a.models for lr in GRID[m]]
    else:
        best = {k: float(v) for k, v in (p.split("=") for p in a.lr)}
        jobs = [(m, best[m], s, f"final_{m}_s{s}") for s in a.seeds for m in a.models]
    for m, lr, s, tag in jobs:
        c = cfg(a, m, lr, s, tag)
        done = Path(c["run_dir"]) / "summary.json"
        if done.exists():
            print("skip", tag, json.loads(done.read_text()), flush=True); continue
        t = time.time(); print("start", tag, flush=True)
        summ = trainer.run(c)
        print("done", tag, summ, f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
