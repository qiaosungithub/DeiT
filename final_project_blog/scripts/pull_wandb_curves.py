#!/usr/bin/env python3
"""Pull W&B histories and render report figures.

This script uses the W&B GraphQL endpoint directly because the installed
`wandb` package in this environment is incompatible with the local protobuf
version. Authentication is read from ~/.netrc, matching W&B's own CLI.
"""

from __future__ import annotations

import json
import netrc
from pathlib import Path

import matplotlib.pyplot as plt
import requests


ENTITY = "sqa24-massachusetts-institute-of-technology"
PROJECT = "deit"
GRAPHQL = "https://api.wandb.ai/graphql"
OUT_DIR = Path(__file__).resolve().parents[1] / "assets"

RUNS = {
    "best_k80_mix": {
        "id": "xluqx5dd",
        "label": "80 mask samples/image forward + mixed labels",
    },
    "k1_low_aug": {
        "id": "cs9p1fu0",
        "label": "K = 1",
    },
    "k10_low_aug": {
        "id": "3ii2ypv2",
        "label": "K = 10",
    },
    "k80_low_aug": {
        "id": "vvmx5jup",
        "label": "K = 80",
    },
}

KEYS = ["ep", "eval_accuracy", "eval_accuracy_iter", "eval_loss"]


def api_key() -> str:
    auth = netrc.netrc().authenticators("api.wandb.ai")
    if not auth:
        raise RuntimeError("No W&B credentials found in ~/.netrc")
    return auth[2]


def fetch_history(run_id: str, samples: int = 1000) -> list[dict]:
    query = """
    query Run($entity: String!, $project: String!, $name: String!, $specs: [JSONString!]!) {
      project(name: $project, entityName: $entity) {
        run(name: $name) {
          name
          displayName
          sampledHistory(specs: $specs)
        }
      }
    }
    """
    spec = json.dumps({"keys": KEYS, "samples": samples})
    variables = {
        "entity": ENTITY,
        "project": PROJECT,
        "name": run_id,
        "specs": [spec],
    }
    resp = requests.post(
        GRAPHQL,
        json={"query": query, "variables": variables},
        auth=("api", api_key()),
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    run = payload["data"]["project"]["run"]
    if run is None:
        raise RuntimeError(f"W&B run not found: {run_id}")
    rows = run["sampledHistory"][0]
    return [
        row for row in rows
        if row.get("ep") is not None and row.get("eval_accuracy_iter") is not None
    ]


def pct(values: list[float]) -> list[float]:
    return [100.0 * v for v in values]


def plot_best_curve(histories: dict[str, list[dict]]) -> None:
    rows = histories["best_k80_mix"]
    ep = [r["ep"] for r in rows]
    single = pct([r["eval_accuracy"] for r in rows])
    iterative = pct([r["eval_accuracy_iter"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ep, single, marker="o", linewidth=2.2, label="Single generated label")
    ax.plot(ep, iterative, marker="o", linewidth=2.2, label="4-step iterative label")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation label accuracy (%)")
    ax.set_title("Masked-bit classifier learning curve")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    ax.set_ylim(0, 84)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "best_training_curve.svg")
    plt.close(fig)


def plot_mask_samples(histories: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key in ["k1_low_aug", "k10_low_aug", "k80_low_aug"]:
        rows = histories[key]
        ep = [r["ep"] for r in rows]
        iterative = pct([r["eval_accuracy_iter"] for r in rows])
        ax.plot(ep, iterative, marker="o", linewidth=2.1, label=RUNS[key]["label"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("4-step validation label accuracy (%)")
    ax.set_title("Effect of mask samples per image forward")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    ax.set_ylim(0, 72)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mask_samples_curve.svg")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = {name: fetch_history(meta["id"]) for name, meta in RUNS.items()}
    (OUT_DIR / "wandb_curves.json").write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n"
    )
    plot_best_curve(histories)
    plot_mask_samples(histories)
    print(f"Wrote figures and raw histories to {OUT_DIR}")


if __name__ == "__main__":
    main()
