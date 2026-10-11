"""Measure a closed-form merge's held-out task-indexed functional residual."""
import json
import os
from pathlib import Path
import sys

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Subset
from transformers import CLIPModel, CLIPVisionModel

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.utils import instantiate


@torch.no_grad()
def main(repo: Path, subset_id: str, merged_path: Path, output_path: Path) -> None:
    device = "cuda"
    with initialize_config_dir(config_dir=str(repo / "config"), version_base=None):
        cfg = compose(
            config_name="model_fusion",
            overrides=[f"modelpool=CLIPVisionModelPool/residual/{subset_id}"],
        )
    modelpool = instantiate(cfg.modelpool)
    processor = modelpool.load_processor()
    projection = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).visual_projection.to(device).eval()
    projection.requires_grad_(False)
    merged = CLIPVisionModel.from_pretrained(merged_path).to(device).eval()

    per_task = {}
    squared_sum = 0.0
    sample_count = 0
    for task in modelpool.model_names:
        raw = modelpool.load_train_dataset(task)
        # RegMean++ uses RandomSampler(seed=0) and exactly the first 256 samples.
        # The following 256 indices are therefore held out from its statistics.
        generator = torch.Generator().manual_seed(0)
        # DataLoader consumes one int64 draw for its iterator base seed before
        # RandomSampler calls randperm with this same generator.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        order = torch.randperm(len(raw), generator=generator)
        indices = order[256 : min(512, len(raw))].tolist()
        dataset = CLIPDataset(Subset(raw, indices), processor)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        expert = modelpool.load_model(task).to(device).eval()
        task_sum = 0.0
        task_count = 0
        for images, _ in loader:
            images = images.to(device)
            merged_z = projection(merged(images)[1])
            expert_z = projection(expert(images)[1])
            merged_z = merged_z / merged_z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            expert_z = expert_z / expert_z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            residual = (merged_z - expert_z).square().sum(dim=-1)
            task_sum += residual.sum().item()
            task_count += residual.numel()
        per_task[task] = {"squared_l2": task_sum / task_count, "n": task_count}
        squared_sum += task_sum
        sample_count += task_count
        del expert
        torch.cuda.empty_cache()

    result = {
        "subset_id": subset_id,
        "tasks": list(modelpool.model_names),
        "heldout_definition": "next 256 indices after RegMean++ seed-0 first 256",
        "space": "normalized CLIP visual_projection output",
        "squared_l2": squared_sum / sample_count,
        "n": sample_count,
        "per_task": per_task,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: measure_residual.py REPO SUBSET_ID MERGED_PATH OUTPUT_JSON"
        )
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
