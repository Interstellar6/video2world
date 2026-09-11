#!/usr/bin/env python3
"""Restore pinned Wan weights at FixAnything's BF16 inference precision without raw shard copies."""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time
from urllib.parse import quote, urlencode

sys.dont_write_bytecode = True
REPOSITORY = "Wan-AI/Wan2.1-I2V-14B-480P"
REVISION = "6b73f84e66371cdfe870c72acd6826e1d61cf279"
GIB = 1024**3
CHUNK = 8 * 1024**2
DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
T5 = "models_t5_umt5-xxl-enc-bf16.pth"
CLIP = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
VAE = "Wan2.1_VAE.pth"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(CHUNK), b""):
            digest.update(data)
    return digest.hexdigest()


def read_exact(stream, size):
    pieces = []
    remaining = size
    while remaining:
        data = stream.read(remaining)
        if not data:
            raise ValueError(f"truncated source stream; {remaining} bytes missing")
        pieces.append(data)
        remaining -= len(data)
    return b"".join(pieces)


def bf16_data(data):
    import torch
    if len(data) % 4:
        raise ValueError("FP32 chunk is not aligned")
    tensor = torch.frombuffer(bytearray(data), dtype=torch.float32)
    return tensor.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()


def converted_header(header, provenance):
    entries = sorted(((key, value) for key, value in header.items() if key != "__metadata__"),
                     key=lambda item: item[1]["data_offsets"])
    converted, input_offset, output_offset = {}, 0, 0
    for key, value in entries:
        dtype, shape = value["dtype"], value["shape"]
        if dtype not in DTYPE_SIZE or not isinstance(shape, list) or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError(f"unsupported tensor type or shape: {key}")
        byte_count = math.prod(shape) * DTYPE_SIZE[dtype]
        if value["data_offsets"] != [input_offset, input_offset + byte_count]:
            raise ValueError(f"non-contiguous or invalid tensor data offsets: {key}")
        output_bytes = byte_count // 2 if dtype == "F32" else byte_count
        converted[key] = {"dtype": "BF16" if dtype == "F32" else dtype, "shape": shape,
                          "data_offsets": [output_offset, output_offset + output_bytes]}
        input_offset += byte_count
        output_offset += output_bytes
    converted["__metadata__"] = {**header.get("__metadata__", {}),
        "conversion": "torch.float32.to(torch.bfloat16); other dtypes preserved",
        "source_repository": REPOSITORY, "source_revision": REVISION, **provenance}
    encoded = json.dumps(converted, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return entries, converted, encoded, input_offset, output_offset


def convert_safetensors(source, destination, expected_sha256, provenance, progress=None):
    digest, source_bytes, output_bytes = hashlib.sha256(), 0, 0

    def consume(count):
        nonlocal source_bytes
        data = read_exact(source, count)
        digest.update(data)
        source_bytes += len(data)
        return data

    header_size = struct.unpack("<Q", consume(8))[0]
    if not 2 <= header_size <= 16 * 1024**2:
        raise ValueError("invalid safetensors header size")
    header = json.loads(consume(header_size))
    entries, output_header, encoded, data_size, output_size = converted_header(header, provenance)
    destination.write(struct.pack("<Q", len(encoded)))
    destination.write(encoded)
    output_bytes += 8 + len(encoded)
    for name, value in entries:
        remaining = value["data_offsets"][1] - value["data_offsets"][0]
        while remaining:
            count = min(CHUNK, remaining)
            data = consume(count)
            result = bf16_data(data) if value["dtype"] == "F32" else data
            destination.write(result)
            output_bytes += len(result)
            remaining -= count
            if progress:
                progress(source_bytes, output_bytes, name)
    if source.read(1):
        raise ValueError("unexpected trailing source bytes")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"source SHA256 mismatch: {actual_sha256} != {expected_sha256}")
    return {"source_sha256": actual_sha256, "source_sha256_verified": True,
            "source_bytes": source_bytes, "output_bytes": output_bytes, "tensor_count": len(entries),
            "source_dtypes": dict(collections.Counter(value["dtype"] for _, value in entries)),
            "output_dtypes": dict(collections.Counter(value["dtype"] for key, value in output_header.items() if key != "__metadata__")),
            "tensor_manifest": {name: {"shape": value["shape"], "source_dtype": value["dtype"],
                                       "output_dtype": output_header[name]["dtype"]} for name, value in entries}}


class RangeReader:
    """A sequential reader that reconnects at the last consumed byte on transient HTTP failures."""
    def __init__(self, url, size, start=0, direct=False):
        self.url, self.size, self.position = url, size, start
        self.response = None
        self.direct = direct

    def _connect(self):
        import requests
        self.close()
        session = requests.Session()
        session.trust_env = not self.direct
        response = session.get(self.url, headers={"Range": f"bytes={self.position}-{self.size - 1}",
                                                  "Accept-Encoding": "identity"}, stream=True, timeout=(30, 90))
        response.raise_for_status()
        valid = response.status_code == 206 and response.headers.get("Content-Range") == f"bytes {self.position}-{self.size - 1}/{self.size}"
        valid = valid or (self.position == 0 and response.status_code == 200 and int(response.headers.get("Content-Length", -1)) == self.size)
        if not valid:
            response.close()
            session.close()
            raise ValueError("server did not honor pinned source byte range")
        self.session = session
        self.response = response

    def read(self, size):
        import requests
        import urllib3
        if self.position >= self.size:
            return b""
        for attempt in range(8):
            try:
                if self.response is None:
                    self._connect()
                data = self.response.raw.read(min(size, self.size - self.position))
                if not data:
                    raise IOError("source HTTP stream ended early")
                self.position += len(data)
                return data
            except (OSError, requests.exceptions.RequestException, urllib3.exceptions.HTTPError) as error:
                self.close()
                if attempt == 7:
                    raise
                print(f"network retry={attempt + 1} offset={self.position} error={type(error).__name__}", flush=True)
                time.sleep(min(2**attempt, 20))
        raise RuntimeError("unreachable")

    def close(self):
        if self.response is not None:
            self.response.close()
            self.session.close()
        self.response = None


class DiskReservedWriter:
    def __init__(self, stream, volume, reserve_bytes):
        self.stream, self.volume, self.reserve_bytes = stream, volume, reserve_bytes

    def write(self, data):
        if shutil.disk_usage(self.volume).free - len(data) < self.reserve_bytes:
            raise ValueError("write would violate the reserved 4 GiB disk space")
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def source_url(filename):
    return f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{quote(filename, safe='/')}"


def source_stream(record):
    return RangeReader(record.get("transport_url", source_url(record["filename"])), record["size"],
                       direct=record.get("transport") == "modelscope-direct")


def validate_mirror_record(record, mirror):
    if mirror.get("Size") != record["size"] or not mirror.get("Sha256") or not mirror.get("Revision"):
        raise ValueError(f"mirror size or identity mismatch: {record['filename']}")
    if record["sha256"] and mirror["Sha256"] != record["sha256"]:
        raise ValueError(f"mirror SHA256 differs from pinned official source: {record['filename']}")
    query = urlencode({"Revision": mirror["Revision"], "FilePath": record["filename"]})
    return {**record, "transport": "modelscope-direct", "transport_revision": mirror["Revision"],
            "transport_sha256": mirror["Sha256"],
            "transport_url": f"https://modelscope.cn/api/v1/models/{REPOSITORY}/repo?{query}"}


def configure_transport(files, transport):
    if transport == "huggingface":
        return files
    import requests
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f"https://modelscope.cn/api/v1/models/{REPOSITORY}/repo/files",
                               params={"Revision": "master", "Recursive": "true"}, timeout=(15, 30))
        response.raise_for_status()
        mirrors = {item["Path"]: item for item in response.json()["Data"]["Files"]}
        configured = []
        for record in files:
            selected = validate_mirror_record(record, mirrors.get(record["filename"], {}))
            if not record["sha256"]:
                data_response = session.get(selected["transport_url"], timeout=(15, 30))
                data_response.raise_for_status()
                data = data_response.content
                if len(data) != record["size"] or hashlib.sha256(data).hexdigest() != selected["transport_sha256"]:
                    raise ValueError(f"mirror small-file checksum mismatch: {record['filename']}")
                digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                verify_source_hash(hashlib.sha256(data).hexdigest(), digest, record)
            configured.append(selected)
    return configured


def source_metadata():
    from huggingface_hub import HfApi
    for attempt in range(4):
        try:
            info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5)
    if info.sha != REVISION:
        raise ValueError("source repository revision differs from pin")
    selected = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if not (name.startswith("diffusion_pytorch_model-") and name.endswith(".safetensors") or
                name in (T5, CLIP, VAE, "config.json") or name.startswith("google/")):
            continue
        lfs = sibling.lfs
        sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        selected.append({"filename": name, "size": sibling.size, "sha256": sha256, "git_blob_id": sibling.blob_id})
    if len([entry for entry in selected if entry["filename"].endswith(".safetensors")]) != 7:
        raise ValueError("expected exactly seven official diffusion shards")
    return selected


def available_memory():
    import psutil
    available = psutil.virtual_memory().available
    limit_file = Path("/sys/fs/cgroup/memory.max")
    usage_file = Path("/sys/fs/cgroup/memory.current")
    if limit_file.is_file() and usage_file.is_file():
        limit = limit_file.read_text().strip()
        if limit != "max":
            available = min(available, int(limit) - int(usage_file.read_text()))
    return available


def plan_files(files):
    plan = []
    for record in files:
        name = record["filename"]
        if name.endswith(".safetensors"):
            stream = source_stream(record)
            try:
                header_size = struct.unpack("<Q", read_exact(stream, 8))[0]
                if not 2 <= header_size <= 16 * 1024**2:
                    raise ValueError("invalid header size")
                header = json.loads(read_exact(stream, header_size))
            finally:
                stream.close()
            entries, _, encoded, raw_bytes, output_bytes = converted_header(header, {"source_sha256": record["sha256"]})
            if raw_bytes + 8 + header_size != record["size"]:
                raise ValueError("source size and tensor layout disagree")
            plan.append({**record, "action": "stream_fp32_to_bf16", "output_bytes_estimate": output_bytes + len(encoded) + 8,
                         "source_dtypes": dict(collections.Counter(value["dtype"] for _, value in entries))})
        elif name == CLIP:
            plan.append({**record, "action": "ram_pth_fp32_to_bf16", "output_bytes_estimate": record["size"] // 2 + 2 * 1024**2})
        else:
            plan.append({**record, "action": "verified_copy", "output_bytes_estimate": record["size"]})
    return sorted(plan, key=lambda item: (item["filename"] != CLIP, item["filename"].endswith(".safetensors"), item["filename"]))


def verify_source_hash(data_sha256, git_sha1, record):
    if record["sha256"]:
        if data_sha256 != record["sha256"]:
            raise ValueError(f"source SHA256 mismatch: {record['filename']}")
    elif record["git_blob_id"] and git_sha1 != record["git_blob_id"]:
        raise ValueError(f"source Git blob hash mismatch: {record['filename']}")


def copy_source(record, output, progress):
    stream = source_stream(record)
    digest = hashlib.sha256()
    git_digest = hashlib.sha1(f"blob {record['size']}\0".encode())
    try:
        while data := stream.read(CHUNK):
            digest.update(data)
            git_digest.update(data)
            output.write(data)
            progress(stream.position, stream.position, None)
    finally:
        stream.close()
    verify_source_hash(digest.hexdigest(), git_digest.hexdigest(), record)
    if record.get("transport_sha256") and digest.hexdigest() != record["transport_sha256"]:
        raise ValueError(f"transport SHA256 mismatch: {record['filename']}")
    return {"source_sha256": digest.hexdigest(), "source_sha256_verified": bool(record["sha256"]),
            "git_blob_verified": bool(not record["sha256"] and record["git_blob_id"]), "source_bytes": record["size"]}


def convert_clip(record, output, progress):
    import torch
    required_memory = 4 * record["size"] + 4 * GIB
    if available_memory() < required_memory:
        raise ValueError(f"CLIP in-memory conversion needs {required_memory} bytes available RAM")
    source = io.BytesIO()
    audit = copy_source(record, source, lambda consumed, written, tensor: progress(consumed, 0, tensor))
    source.seek(0)
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("unexpected CLIP checkpoint format")
    audit["tensor_manifest"] = {name: {"shape": list(value.shape), "source_dtype": str(value.dtype),
        "output_dtype": "torch.bfloat16" if value.dtype == torch.float32 else str(value.dtype)} for name, value in state.items()}
    source.close()
    for name in state:
        if state[name].dtype == torch.float32:
            state[name] = state[name].to(torch.bfloat16)
    torch.save(state, output)
    audit["tensor_count"] = len(state)
    return audit


def existing_receipt(target, receipt_path, record):
    if not target.exists():
        return None
    if not receipt_path.is_file():
        raise ValueError(f"existing weights have no acquisition receipt: {target}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("source_revision") != REVISION or (record["sha256"] and receipt.get("source_sha256") != record["sha256"]):
        raise ValueError(f"existing acquisition identity differs: {target}")
    if file_hash(target) != receipt.get("output_sha256"):
        raise ValueError(f"existing acquired weights hash differs: {target}")
    if receipt.get("status") == "prepared":
        receipt["status"] = "complete"
        write_json(receipt_path, receipt)
    return receipt


def main():
    app = argparse.ArgumentParser(description=__doc__)
    app.add_argument("--model-dir", type=Path, required=True)
    app.add_argument("--receipt-dir", type=Path, required=True)
    app.add_argument("--reserve-gib", type=float, default=4)
    app.add_argument("--execute", action="store_true")
    app.add_argument("--transport", choices=("huggingface", "modelscope-direct"), default="huggingface")
    app.add_argument("--predecessor-pid", type=int)
    args = app.parse_args()
    project = Path(__file__).resolve().parents[1]
    model_dir, receipt_dir = args.model_dir.resolve(), args.receipt_dir.resolve()
    if project not in model_dir.parents or (project / "outputs") not in receipt_dir.parents:
        app.error("weights must remain in this project; receipts must remain under project outputs")
    if args.reserve_gib < 4:
        app.error("minimum reserve is 4 GiB")
    model_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    lock = (receipt_dir / "acquisition.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.predecessor_pid:
        import psutil
        if psutil.pid_exists(args.predecessor_pid) and psutil.Process(args.predecessor_pid).status() != psutil.STATUS_ZOMBIE:
            raise ValueError("predecessor download process is still live")
    old_progress_path = receipt_dir / "progress.json"
    predecessor_progress = json.loads(old_progress_path.read_text()) if args.predecessor_pid and old_progress_path.is_file() else None
    plan = plan_files(configure_transport(source_metadata(), args.transport))
    target_root = model_dir / REPOSITORY
    existing = {}
    previous_plan_path = receipt_dir / "acquisition-plan.json"
    previous_plan = json.loads(previous_plan_path.read_text()) if previous_plan_path.is_file() else {}
    previous_files = {item["filename"]: item for item in previous_plan.get("files", [])}
    partials = []
    for record in plan:
        target = target_root / record["filename"]
        if model_dir not in target.resolve().parents:
            raise ValueError("model path resolves outside the configured asset directory")
        receipt = existing_receipt(target, receipt_dir / (record["filename"].replace("/", "_") + ".receipt.json"), record)
        if receipt:
            existing[record["filename"]] = receipt
        partial = target.with_name(target.name + ".acquiring.partial")
        if partial.exists():
            previous = previous_files.get(record["filename"], {})
            if partial.is_symlink() or previous_plan.get("source_revision") != REVISION or previous.get("sha256") != record["sha256"]:
                raise ValueError(f"unowned partial weight file: {partial}")
            partials.append(partial)
    remaining = sum(record["output_bytes_estimate"] for record in plan if record["filename"] not in existing)
    free = shutil.disk_usage(model_dir).free
    reclaimable = sum(path.stat().st_size for path in partials)
    report = {"source_repository": REPOSITORY, "source_revision": REVISION, "files": plan,
              "estimated_output_bytes": sum(record["output_bytes_estimate"] for record in plan), "remaining_output_bytes": remaining,
              "disk_free_bytes": free, "minimum_reserve_bytes": int(args.reserve_gib * GIB),
              "projected_remaining_bytes": free + reclaimable - remaining, "reclaimable_partial_bytes": reclaimable,
              "completed_files": list(existing),
              "transport": args.transport, "predecessor_pid": args.predecessor_pid,
              "predecessor_progress": predecessor_progress,
              "compatibility": "FixAnything loads all four components through model.to(bfloat16); names/shapes retained, VAE and T5 copied unchanged",
              "resume": "completed shards verified and reused; interrupted shard streams restart without a raw disk copy"}
    write_json(receipt_dir / "acquisition-plan.json", report)
    print(json.dumps({key: report[key] for key in ("estimated_output_bytes", "remaining_output_bytes", "disk_free_bytes", "projected_remaining_bytes")}), flush=True)
    if free + reclaimable - remaining < args.reserve_gib * GIB:
        raise ValueError("projected completion would violate the 4 GiB disk reserve")
    if not args.execute:
        return 0
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    for partial in partials:
        with partial.open("wb"):
            pass
    started = time.time()
    for index, record in enumerate(plan):
        name = record["filename"]
        if name in existing:
            continue
        target = target_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".acquiring.partial")
        progress_path = receipt_dir / "progress.json"
        last_report = 0

        def progress(source_bytes, output_bytes, tensor):
            nonlocal last_report
            if time.time() - last_report < 5:
                return
            last_report = time.time()
            disk = shutil.disk_usage(model_dir).free
            if disk < args.reserve_gib * GIB:
                raise ValueError("live free space dropped below reserved 4 GiB")
            write_json(progress_path, {"status": "running", "pid": os.getpid(), "file": name, "file_index": index,
                "file_count": len(plan), "source_bytes_read": source_bytes, "source_bytes_total": record["size"],
                "output_bytes_written": output_bytes, "tensor": tensor, "completed_files": list(existing),
                "elapsed_seconds": time.time() - started, "disk_free_bytes": disk, "updated_unix": time.time()})

        print(f"start {name} action={record['action']}", flush=True)
        with temporary.open("wb") as raw_output:
            output = DiskReservedWriter(raw_output, model_dir, int(args.reserve_gib * GIB))
            if record["action"] == "ram_pth_fp32_to_bf16":
                audit = convert_clip(record, output, progress)
            elif record["action"] == "stream_fp32_to_bf16":
                source = source_stream(record)
                try:
                    audit = convert_safetensors(source, output, record["sha256"], {"source_sha256": record["sha256"]}, progress)
                finally:
                    source.close()
            else:
                audit = copy_source(record, output, progress)
            output.flush()
            os.fsync(output.fileno())
        if record["action"] == "stream_fp32_to_bf16":
            from safetensors import safe_open
            with safe_open(str(temporary), framework="pt", device="cpu") as stored:
                if set(stored.keys()) != set(audit["tensor_manifest"]):
                    raise ValueError("converted tensor key mismatch")
                for key in stored.keys():
                    if list(stored.get_slice(key).get_shape()) != audit["tensor_manifest"][key]["shape"]:
                        raise ValueError(f"converted tensor shape mismatch: {key}")
        audit.update({"status": "prepared", "source_repository": REPOSITORY, "source_revision": REVISION,
            "filename": name, "action": record["action"], "output_bytes": temporary.stat().st_size,
            "output_sha256": file_hash(temporary), "completed_unix": time.time(),
            "transport": record.get("transport", "huggingface"), "transport_url": record.get("transport_url", source_url(name)),
            "transport_revision": record.get("transport_revision", REVISION),
            "transport_sha256": record.get("transport_sha256", record["sha256"])})
        if target.exists():
            raise ValueError(f"target appeared during acquisition: {target}")
        receipt_path = receipt_dir / (name.replace("/", "_") + ".receipt.json")
        write_json(receipt_path, audit)
        temporary.replace(target)
        audit["status"] = "complete"
        write_json(receipt_path, audit)
        existing[name] = audit
        print(f"complete {name} output_bytes={audit['output_bytes']} output_sha256={audit['output_sha256']}", flush=True)
    write_json(receipt_dir / "progress.json", {"status": "complete", "pid": os.getpid(), "source_revision": REVISION,
        "completed_files": list(existing), "output_bytes": sum(record["output_bytes"] for record in existing.values()),
        "elapsed_seconds": time.time() - started, "disk_free_bytes": shutil.disk_usage(model_dir).free})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
