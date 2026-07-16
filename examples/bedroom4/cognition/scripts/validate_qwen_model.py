#!/usr/bin/env python3
"""Validate a local Qwen2.5-VL snapshot before description generation."""

from __future__ import annotations

import argparse
import json
import struct
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from safetensors import safe_open
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

EXPECTED_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()

    required = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (model_dir / name).is_file()]
    incomplete = sorted(str(path) for path in model_dir.rglob("*.incomplete"))
    if missing or incomplete:
        raise RuntimeError(f"Incomplete model: missing={missing}, incomplete={incomplete}")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    shards = []
    all_shard_keys: set[str] = set()
    payload_total = 0
    for name in shard_names:
        path = model_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Missing or empty shard: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        with path.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0] + 8
        payload_bytes = path.stat().st_size - header_bytes
        payload_total += payload_bytes
        expected_keys = {key for key, shard in index["weight_map"].items() if shard == name}
        if set(keys) != expected_keys:
            raise RuntimeError(
                f"Shard/index key mismatch for {name}: missing={sorted(expected_keys - set(keys))[:5]}, "
                f"extra={sorted(set(keys) - expected_keys)[:5]}"
            )
        all_shard_keys.update(keys)
        shards.append(
            {
                "name": name,
                "physical_bytes": path.stat().st_size,
                "header_bytes": header_bytes,
                "tensor_payload_bytes": payload_bytes,
                "tensor_count": len(keys),
            }
        )
    if all_shard_keys != set(index["weight_map"]):
        raise RuntimeError("Safetensors shard union does not equal model index")
    if payload_total != index.get("metadata", {}).get("total_size"):
        raise RuntimeError(
            f"Safetensors payload/index size mismatch: payload={payload_total}, "
            f"index={index.get('metadata', {}).get('total_size')}"
        )

    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    revision_records = []
    for name in [*required, *shard_names]:
        metadata_path = metadata_dir / f"{name}.metadata"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing Hugging Face local-dir metadata: {metadata_path}")
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        revision = lines[0].strip() if lines else ""
        if revision != EXPECTED_REVISION:
            raise RuntimeError(f"Revision mismatch for {name}: {revision}")
        revision_records.append({"file": name, "revision": revision})

    if config.get("model_type") != "qwen2_5_vl":
        raise RuntimeError(f"Unexpected model_type: {config.get('model_type')}")
    if "Qwen2_5_VLForConditionalGeneration" not in (config.get("architectures") or []):
        raise RuntimeError(f"Unexpected architectures: {config.get('architectures')}")

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model-load validation")
    before = torch.cuda.memory_allocated(0)
    model, loading_info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    loading_issues = {
        key: loading_info.get(key) or []
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if any(loading_issues.values()):
        raise RuntimeError(f"Model load reported key issues: {loading_issues}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 3_500_000_000 <= parameter_count <= 4_000_000_000:
        raise RuntimeError(f"Unexpected parameter count: {parameter_count}")
    after = torch.cuda.memory_allocated(0)
    result = {
        "schema_version": 1,
        "validated_at": datetime.now().astimezone().isoformat(),
        "status": "passed",
        "model_dir": str(model_dir),
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "revision": EXPECTED_REVISION,
        "revision_evidence": revision_records,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "index_total_size": index.get("metadata", {}).get("total_size"),
        "shards": shards,
        "parameter_count": parameter_count,
        "processor_class": processor.__class__.__name__,
        "image_processor_class": processor.image_processor.__class__.__name__,
        "use_fast_image_processor": False,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "device": torch.cuda.get_device_name(0),
        "cuda_memory_allocated_before": before,
        "cuda_memory_allocated_after_load": after,
        "local_files_only": True,
        "loading_info": loading_issues,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
