#!/usr/bin/env python3
"""Record Python/library/GPU versions and the Git revision to a JSON file (run at every session start)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plantrel.repro import write_environment_log  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--config", help="optional YAML/JSON config to record and hash")
    a = p.parse_args()
    cfg = None
    if a.config:
        text = Path(a.config).read_text(encoding="utf-8")
        if a.config.endswith(".json"):
            cfg = json.loads(text)
        else:
            import yaml
            cfg = yaml.safe_load(text)
    rec = write_environment_log(a.out, config=cfg, repo_dir=Path(__file__).resolve().parents[1])
    env = rec["environment"]
    print(f"python {env['python']} | torch {env['packages'].get('torch')} | cuda_available "
          f"{env['cuda'].get('cuda_available')} | gpus {env['cuda'].get('gpus')} | git {env['git']['commit']}")
    if not env["cuda"].get("cuda_available"):
        print("NOTE: no GPU visible. Do not start training; evaluation and calibration can run on CPU.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
