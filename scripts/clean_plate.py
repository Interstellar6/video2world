#!/usr/bin/env python3
"""Fixed-camera, mask-limited clean plates through Responses image_generation.

The request budget is cumulative for this task's clean_plate stage. A submitted
request is never automatically resubmitted, even after a timeout or process crash.
Outputs remain generated candidates, not observations or accepted scene geometry.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.provider_io import input_artifact, local_path, parser as provider_parser, publish, read_json


class CleanPlateError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CleanPlateError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def relative(task, path):
    return local_path(task, path).relative_to(task.resolve()).as_posix()


def atomic_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode())


def save_png(path, image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    atomic_bytes(path, buffer.getvalue())


def file_record(task, path):
    return {"path": relative(task, path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value), "unsafe frame or object identifier")
    return value


def indexed(records, key, label):
    require(isinstance(records, list) and all(isinstance(item, dict) for item in records), f"{label} needs a records list")
    result = {}
    for record in records:
        name = identifier(record.get(key))
        require(name not in result, f"duplicate {label}: {name}")
        result[name] = record
    return result


def rgb(path, size=None):
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        require(image.getexif().get(274, 1) == 1, "source has nontrivial EXIF orientation; calibrate the stored raster first")
        require(size is None or image.size == size, "image dimensions differ from the calibrated source raster")
        return image.convert("RGB")


def binary_mask(task, record, size):
    import numpy as np
    from PIL import Image

    require(record.get("mask_coordinate_frame", "source_image") in {"source_image", "original_image"}, "mask must use the original calibrated source raster")
    path = local_path(task, record["mask_path"])
    if record.get("sha256"):
        require(sha256(path) == record["sha256"], "mask hash mismatch")
    with Image.open(path) as image:
        require(image.size == size and image.mode in {"1", "L"}, "mask must be a single-channel original-resolution image")
        values = np.asarray(image.convert("L"))
    require(np.isin(values, [0, 255]).all(), "mask must be binary 0/255")
    return values > 0, path


def load_plan(args):
    import numpy as np
    from PIL import Image, ImageFilter

    task = args.task_dir.resolve()
    roles = ("frames_manifest", "component_masks", "physical_instance_tracks", "lifting_report")
    envelope = read_json(local_path(task, args.inputs))["inputs"]
    paths = {role: input_artifact(args, role) for role in roles}
    for role, path in paths.items():
        require(envelope[role].get("status") not in {"failed", "blocked", "stale", "contract_only"}, f"{role} is not available")
        if envelope[role].get("sha256"):
            require(sha256(path) == envelope[role]["sha256"], f"{role} input hash mismatch")
    documents = {role: read_json(path) for role, path in paths.items()}
    lifting = documents["lifting_report"]
    require(lifting.get("status") in {"passed", "partial_tracks_rejected"} and lifting.get("carve_performed") is True, "lifting has not accepted and carved physical objects")
    require(lifting.get("generated_geometry_used") is False, "clean plate removal requires observed lifting evidence")
    accepted = lifting.get("accepted_object_ids")
    require(isinstance(accepted, list) and accepted and len(set(accepted)) == len(accepted), "lifting must name unique accepted objects")
    reports = indexed(lifting.get("tracks"), "object_id", "lifting track")
    require(set(accepted) == {name for name, record in reports.items() if record.get("accepted") is True}, "lifting accepted-object declarations disagree")
    for role in ("component_masks", "physical_instance_tracks"):
        binding = lifting.get("resolved_files", lifting.get("source_files", {})).get(role, {})
        require(binding.get("sha256") == sha256(paths[role]) and local_path(task, binding.get("path", "")) == paths[role], f"lifting does not bind current {role}")
    frames = indexed(documents["frames_manifest"].get("frames"), "frame_id", "source frame")
    require(frames, "source frames cannot be empty")
    tracks = indexed(documents["physical_instance_tracks"].get("tracks"), "object_id", "physical track")
    require(set(accepted) <= tracks.keys(), "accepted object missing from physical tracks")
    masks = documents["component_masks"].get("masks")
    require(isinstance(masks, list), "component_masks requires masks")
    parent_masks = {}
    for item in masks:
        if item.get("component_id") != "__object__":
            continue
        key = (identifier(item.get("object_id")), identifier(item.get("frame_id")))
        require(key not in parent_masks, "duplicate parent mask")
        require(key[1] in frames, "parent mask references an unknown source frame")
        parent_masks[key] = item
    selected = {}
    for name in accepted:
        observations = indexed(tracks[name].get("observations"), "frame_id", "track observation")
        require(observations, "accepted track has no observations")
        for frame_id, observation in observations.items():
            key = (name, frame_id)
            require(key in parent_masks, "accepted observation has no matching parent mask")
            parent = parent_masks[key]
            require(local_path(task, observation["mask_path"]) == local_path(task, parent["mask_path"]), "track and parent mask paths disagree")
            selected[key] = parent
    jobs, dependencies = [], {role: file_record(task, path) for role, path in paths.items()}
    for frame_id, frame in frames.items():
        source = local_path(task, frame["image_path"])
        size = (frame.get("width"), frame.get("height"))
        require(all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size), "source dimensions must be positive integers")
        rgb(source, size)
        union = np.zeros(size[::-1], dtype=bool)
        protected = np.zeros_like(union)
        object_ids, bindings = [], []
        for (name, mask_frame), record in parent_masks.items():
            if mask_frame != frame_id:
                continue
            mask, path = binary_mask(task, record, size)
            if record.get("image_path"):
                require(local_path(task, record["image_path"]) == source, "parent mask is not bound to the source image")
            bindings.append({"object_id": name, **file_record(task, path)})
            if (name, frame_id) in selected:
                require(mask.any(), "accepted parent mask is empty")
                union |= mask
                object_ids.append(name)
            else:
                protected |= mask
        # Ownership resolution: disputed boundary pixels belong to the protected
        # (unaccepted) object. A small shared border is normal when two masks meet;
        # a large overlap still fails because it would silently change ownership.
        disputed = int((union & protected).sum())
        require(disputed <= args.max_ownership_overlap_fraction * max(int(union.sum()), 1),
                f"{frame_id}: accepted mask overlaps an unaccepted object by {disputed} pixels; resolve ownership before generation")
        union &= ~protected
        initial_pixels = int(union.sum())
        if args.dilate_pixels and union.any():
            union = np.asarray(Image.fromarray(union.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * args.dilate_pixels + 1))) > 0
        protected_pixels = int((union & protected).sum())
        union &= ~protected
        require(float(union.mean()) <= args.max_mask_fraction, f"{frame_id}: removal mask is too large for fixed-camera completion")
        # Unsegmented source views could train the removed object back into PGSR.
        # Only keyframes with a validated removal mask become clean-plate inputs.
        if not object_ids:
            continue
        jobs.append({"frame_id": frame_id, "source": source, "width": size[0], "height": size[1], "mask": union,
                     "object_ids": sorted(object_ids), "source_binding": file_record(task, source), "parent_masks": bindings,
                     "parent_pixels": initial_pixels, "masked_pixels": int(union.sum()), "protected_dilation_pixels": protected_pixels,
                     "protected_ownership_pixels": disputed})
    require(any(job["masked_pixels"] for job in jobs), "accepted tracks have no editable source pixels")
    return jobs, dependencies, sorted(accepted)


def validate_options(args):
    local_model = getattr(args, "local_lama_model", None)
    if local_model is not None:
        require(Path(local_model).is_file(), "local LaMa ONNX model is missing")
        require(args.controller and args.model, "controller and image model must be explicit")
        require(0 <= args.dilate_pixels <= 32 and 0 < args.max_mask_fraction < 1 and 0 <= args.max_ownership_overlap_fraction < 1, "dilation must be 0..32, mask fraction in (0,1) and ownership overlap in [0,1)")
        require(args.max_api_requests >= 0 and 0 < args.timeout_seconds <= 3600, "invalid API request budget or timeout")
        require(0 <= args.max_raw_outside_mae <= 255 and 0 <= args.max_aspect_error <= 0.1, "invalid generated raster QA thresholds")
        return
    parsed = urllib.parse.urlsplit(args.endpoint)
    local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    require((parsed.scheme == "https" or local_http) and parsed.hostname and parsed.path.rstrip("/").endswith("/responses"), "endpoint must be the complete HTTPS Responses URL (or loopback HTTP for tests)")
    require(not parsed.username and not parsed.password and not parsed.query and not parsed.fragment, "endpoint cannot contain credentials, query or fragment")
    require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.token_env), "invalid credential environment-variable name")
    require(args.controller and args.model, "controller and image model must be explicit")
    require(args.size in {"source", "auto"} or re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", args.size), "size must be source, auto, or WIDTHxHEIGHT")
    require(0 <= args.dilate_pixels <= 32 and 0 < args.max_mask_fraction < 1, "dilation must be 0..32 and mask fraction in (0,1)")
    require(args.max_api_requests >= 0 and 0 < args.timeout_seconds <= 3600, "invalid API request budget or timeout")
    require(0 <= args.max_raw_outside_mae <= 255 and 0 <= args.max_aspect_error <= 0.1, "invalid generated raster QA thresholds")


def settings(args):
    names = ("endpoint", "controller", "model", "reasoning_effort", "quality", "size", "dilate_pixels", "max_mask_fraction", "max_raw_outside_mae", "max_aspect_error")
    return {name: getattr(args, name) for name in names}


def prompt_for(job, anchor):
    lines = [
        "Use case: precise-object-edit. Purpose: synthetic clean plate for candidate background reconstruction, not observed evidence.",
        f"Image 1 is the complete edit target, frame {job['frame_id']}, calibrated raster {job['width']}x{job['height']}.",
        "Image 2 is a binary ownership mask: WHITE is the only editable region; BLACK must stay unchanged.",
        "Remove only the physical objects within the white mask, including their masked remnants. Fill with photorealistic background continuation inferred from visible walls, floor, surfaces, perspective and existing illumination.",
        "Preserve the target camera, projection, viewpoint, field of view, canvas bounds, structural edges, all unmasked objects, unmasked shadows, exposure, white balance and every unmasked pixel. No crop, zoom, reframing, camera movement, relighting, added furniture, decorations, text or watermark.",
        "Do not replace the image with another view. Do not render the mask. Match edge texture, avoid seams, repeated patterns, flat black fills, ghost silhouettes or invented structure where visible evidence constrains the continuation.",
        f"Return exactly one PNG of the entire target at {job['width']}x{job['height']}. Object IDs in the editable mask: {', '.join(job['object_ids'])}.",
    ]
    if anchor:
        lines += [
            f"Image 3 is a previously generated clean-plate anchor from frame {anchor['frame_id']}, supplied ONLY as a cross-view material/structure consistency reference.",
            "The anchor is a synthetic candidate, not ground truth. Keep the target viewpoint, do not copy its camera or paste its pixels; reconcile shared background appearance with visible target geometry.",
        ]
    return "\n".join(lines)


def data_url(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_request(args, job, source, mask, anchor):
    requested_size = f"{job['width']}x{job['height']}" if args.size == "source" else args.size
    if args.model == "gpt-image-2" and requested_size != "auto":
        width, height = map(int, requested_size.split("x"))
        require(width % 16 == height % 16 == 0 and max(width, height) <= 3840 and max(width, height) / min(width, height) <= 3
                and 655360 <= width * height <= 8294400, "gpt-image-2 requested size violates model raster limits; choose an explicit compatible same-aspect size")
    content = [{"type": "input_text", "text": prompt_for(job, anchor)},
               {"type": "input_image", "image_url": data_url(source), "detail": "high"},
               {"type": "input_image", "image_url": data_url(mask), "detail": "high"}]
    if anchor:
        content.append({"type": "input_image", "image_url": data_url(anchor["path"]), "detail": "high"})
    return {"model": args.controller, "reasoning": {"effort": args.reasoning_effort}, "store": False,
            "input": [{"role": "user", "content": content}],
            "tools": [{"type": "image_generation", "model": args.model, "action": "edit", "quality": args.quality,
                       "size": requested_size, "output_format": "png"}], "tool_choice": {"type": "image_generation"}}


def local_lama_transport(model_path):
    """Offline LaMa inpainting behind the Responses ABI: no credential, no network.

    The adapter's journal, budget, compositing and QA logic are reused unchanged;
    only the transport is local. The model is the official OpenCV LaMa ONNX asset.
    """
    import numpy as np
    import cv2
    from PIL import Image

    model = cv2.dnn.readNetFromONNX(str(model_path))

    def decode(value):
        require(isinstance(value, str) and value.startswith("data:image/"), "local inpainting requires data-URL images")
        try:
            payload = base64.b64decode(value.split(",", 1)[1], validate=True)
        except (ValueError, TypeError) as error:
            raise CleanPlateError("local inpainting received invalid base64") from error
        with Image.open(BytesIO(payload)) as image:
            image.load()
            return image.convert("RGB")

    def transport(endpoint, token, payload, timeout, request_id):
        content = payload["input"][0]["content"]
        urls = [item.get("image_url") for item in content if item.get("type") == "input_image"]
        require(len(urls) >= 2, "local inpainting requires a source image and a binary ownership mask")
        source, mask = decode(urls[0]), decode(urls[1])
        rgb = np.asarray(source)
        binary = (np.asarray(mask.convert("L")) > 127).astype(np.float32)
        image_blob = cv2.dnn.blobFromImage(rgb[:, :, ::-1].copy(), 0.00392, (512, 512), (0, 0, 0), False, False)
        mask_blob = cv2.dnn.blobFromImage(binary, scalefactor=1.0, size=(512, 512), mean=(0,), swapRB=False, crop=False)
        model.setInput(image_blob, "image")
        model.setInput((mask_blob > 0).astype(np.float32), "mask")
        output = np.transpose(model.forward()[0], (1, 2, 0)).astype(np.uint8)
        result = cv2.resize(output, source.size)[:, :, ::-1]
        buffer = BytesIO()
        Image.fromarray(result).save(buffer, format="PNG")
        body = json.dumps({"status": "completed", "output": [
            {"type": "image_generation_call", "result": base64.b64encode(buffer.getvalue()).decode("ascii")}]}).encode()
        return 200, {}, body

    return transport


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def post_response(endpoint, token, payload, timeout, request_id):
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/json", "X-Client-Request-Id": request_id})
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


@contextmanager
def stage_lock(task, stage):
    path = local_path(task, stage / ".provider.lock", exists=False)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CleanPlateError("another clean_plate provider owns the task lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def extract_png(response):
    require(response.get("status") in {None, "completed"}, "Responses result is not completed")
    images = [item for item in response.get("output", []) if item.get("type") == "image_generation_call" and item.get("result")]
    require(len(images) == 1, "Responses must contain exactly one image_generation_call result")
    try:
        value = base64.b64decode(images[0]["result"], validate=True)
    except (ValueError, TypeError) as error:
        raise CleanPlateError("Responses image result is invalid base64") from error
    from PIL import Image
    with Image.open(BytesIO(value)) as image:
        image.load()
        require(image.format == "PNG" and image.width > 0 and image.height > 0, "image_generation result is not a readable PNG")
    return value


def request_once(args, task, directory, request, journal, journal_path, transport):
    request_hash = digest_json({"endpoint": args.endpoint, "request": request})
    entries = journal["requests"]
    existing = next((item for item in reversed(entries) if item["request_sha256"] == request_hash), None)
    if existing is not None:
        if existing["state"] == "response_received":
            raw = local_path(task, existing["response_path"])
            require(sha256(raw) == existing["response_sha256"], "cached response hash mismatch; refusing a new API request")
            return read_json(raw), existing
        require(existing["request_id"] in args.retry_request_id, "previous request failed or has unknown outcome; automatic resubmission is forbidden; inspect api_journal.json and explicitly authorize its request ID to retry")
    require(len(entries) < args.max_api_requests, "cumulative API request budget exhausted; increase --max-api-requests explicitly to continue")
    token = "local-inpainting-no-credential" if getattr(args, "local_lama_model", None) is not None else os.environ.get(args.token_env)
    require(token and not any(character in token for character in "\r\n"), f"credential variable {args.token_env} is absent or invalid")
    entry = {"request_id": str(uuid.uuid4()), "request_sha256": request_hash, "state": "submitted_outcome_unknown",
             "submitted_at": datetime.now(timezone.utc).isoformat(), "endpoint": args.endpoint,
             "frame_dir": relative(task, directory), "automatic_retry_allowed": False}
    if existing:
        entry["explicit_retry_of_request_id"] = existing["request_id"]
    entries.append(entry)
    write_json(journal_path, journal)
    raw_path = local_path(task, directory / "rawresponse" / (entry["request_id"] + ".json"), exists=False)
    try:
        status, headers, body = transport(args.endpoint, token, request, args.timeout_seconds, entry["request_id"])
    except Exception as error:
        entry["failure_type"] = type(error).__name__
        write_json(journal_path, journal)
        raise CleanPlateError("Responses transport failed; request outcome and possible charge are unknown; not retried") from None
    # Never serialize authorization headers or a credential echoed by a provider.
    sanitized = body.replace(token.encode(), b"[REDACTED]")
    atomic_bytes(raw_path, sanitized)
    entry.update({"http_status": status, "response_path": relative(task, raw_path), "response_sha256": sha256(raw_path),
                  "provider_request_id": next((value.replace(token, "[REDACTED]") for key, value in headers.items() if key.lower() == "x-request-id"), None),
                  "response_received_at": datetime.now(timezone.utc).isoformat(), "credential_redacted": sanitized != body})
    entry["state"] = "response_received" if 200 <= status < 300 else "http_error_no_retry"
    write_json(journal_path, journal)
    require(200 <= status < 300, f"Responses HTTP {status}; no automatic retry; inspect rawresponse and api_journal.json")
    try:
        response = read_json(raw_path)
    except (ValueError, UnicodeError) as error:
        raise CleanPlateError("Responses returned invalid JSON; raw response retained; not retried") from error
    return response, entry


def composite(args, raw, source, mask, output):
    import numpy as np
    from PIL import Image

    original = rgb(source)
    with Image.open(raw) as generated:
        generated.load()
        require(generated.mode in {"RGB", "RGBA"}, "generated clean plate must be RGB or opaque RGBA")
        if generated.mode == "RGBA":
            require(generated.getchannel("A").getextrema() == (255, 255), "generated clean plate contains transparency")
        raw_size = generated.size
        aspect_error = abs((generated.width / generated.height) / (original.width / original.height) - 1)
        require(aspect_error <= args.max_aspect_error, "generated image changed aspect ratio; fixed-camera normalization rejected")
        aligned = generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    with Image.open(mask) as mask_image:
        selected = np.asarray(mask_image.convert("L")) > 0
    old, new = np.asarray(original), np.asarray(aligned)
    difference = np.abs(new.astype(np.float32) - old.astype(np.float32))
    outside_mae = float(difference[~selected].mean()) if (~selected).any() else 0.0
    require(outside_mae <= args.max_raw_outside_mae, "raw generated image drifted outside the mask; candidate rejected before hard composition")
    changed = np.any(new != old, axis=2) & selected
    require(changed.any(), "generated mask pixels are unchanged; source-copy is not a clean plate")
    output_pixels = old.copy()
    output_pixels[selected] = new[selected]
    save_png(output, Image.fromarray(output_pixels))
    written = np.asarray(rgb(output, original.size))
    require(np.array_equal(written[~selected], old[~selected]), "outside-mask preservation failed after PNG roundtrip")
    return {"raw_dimensions": list(raw_size), "final_dimensions": list(original.size), "raw_aspect_error": aspect_error,
            "raw_to_source": [[original.width / raw_size[0], 0, 0], [0, original.height / raw_size[1], 0], [0, 0, 1]],
            "resampling": "lanczos" if raw_size != original.size else "identity", "raw_outside_mask_mae": outside_mae,
            "outside_mask_changed_pixels": 0, "changed_mask_pixels": int(changed.sum()), "masked_pixels": int(selected.sum()),
            "visual_review": "pending", "cross_view_geometry_validation": "not_performed", "promotion_allowed": False}


def verify_record(task, record):
    path = local_path(task, record["path"])
    require(path.is_file() and sha256(path) == record["sha256"], "cached clean-plate artifact hash mismatch")
    return path


def write_preview(path, source, mask, raw, final):
    from PIL import Image, ImageOps

    panels = []
    for value in (source, mask, raw, final):
        with Image.open(value) as image:
            panels.append(ImageOps.contain(image.convert("RGB"), (640, 360)))
    canvas = Image.new("RGB", (1280, 720), (255, 255, 255))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 2) * 640, (index // 2) * 360))
    save_png(path, canvas)


def execute_locked(args, task, stage, jobs, dependencies, accepted, transport):
    from PIL import Image
    import numpy as np

    recipe = {"schema_version": "1.0", "adapter_sha256": sha256(Path(__file__)), "settings": settings(args), "inputs": dependencies,
              "frames": [{key: value for key, value in job.items() if key not in {"mask", "source"}} for job in jobs]}
    recipe_id = digest_json(recipe)
    run_dir = local_path(task, stage / "runs" / recipe_id, exists=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "recipe.json", recipe)
    journal_path = local_path(task, stage / "api_journal.json", exists=False)
    journal = read_json(journal_path) if journal_path.exists() else {"schema_version": "1.0", "stage": "clean_plate", "requests": []}
    require(journal.get("stage") == "clean_plate" and isinstance(journal.get("requests"), list), "invalid clean_plate request journal")
    report_path = run_dir / "clean_plate_report.json"
    output_frames = []
    all_frames = read_json(local_path(task, dependencies["frames_manifest"]["path"]))["frames"]
    selected_ids = {job["frame_id"] for job in jobs}
    report = {"schema_version": "1.0", "status": "running", "evidence": "generated", "camera_policy": "fixed_source_camera",
              "recipe_id": recipe_id, "accepted_object_ids": accepted, "source_artifacts": dependencies,
              "total_source_frames": len(all_frames), "planned_api_frames": sum(job["masked_pixels"] > 0 for job in jobs),
              "excluded_without_accepted_removal_mask": [item["frame_id"] for item in all_frames if item["frame_id"] not in selected_ids],
              "frames": output_frames, "api_journal_path": relative(task, journal_path) if journal_path.exists() else journal_path.relative_to(task).as_posix(),
              "promotion_allowed": False, "collision_eligible": False, "visual_review": "pending",
              "cross_view_geometry_validation": "not_performed", "reconstruction_policy": "generated_candidates_only"}
    anchor = None
    try:
        for job in jobs:
            directory = local_path(task, run_dir / "frames" / job["frame_id"], exists=False)
            directory.mkdir(parents=True, exist_ok=True)
            source = local_path(task, directory / "source/source.png", exists=False)
            mask = local_path(task, directory / "mask/removal.png", exists=False)
            final = local_path(task, directory / "final/clean_plate.png", exists=False)
            receipt_path = local_path(task, directory / "receipt.json", exists=False)
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                require(receipt.get("recipe_id") == recipe_id, "cached frame belongs to another recipe")
                require(receipt.get("source_binding") == job["source_binding"], "cached source changed")
                for record in receipt["artifacts"].values():
                    verify_record(task, record)
            else:
                save_png(source, rgb(job["source"], (job["width"], job["height"])))
                save_png(mask, Image.fromarray(job["mask"].astype(np.uint8) * 255))
                artifacts = {"source": file_record(task, source), "mask": file_record(task, mask)}
                if job["masked_pixels"]:
                    request = build_request(args, job, source, mask, anchor)
                    prompt_path = local_path(task, directory / "prompt/prompt.txt", exists=False)
                    request_path = local_path(task, directory / "prompt/request.json", exists=False)
                    atomic_bytes(prompt_path, prompt_for(job, anchor).encode())
                    write_json(request_path, request)
                    response, api_receipt = request_once(args, task, directory, request, journal, journal_path, transport)
                    raw = local_path(task, directory / "generated/raw.png", exists=False)
                    atomic_bytes(raw, extract_png(response))
                    qa = composite(args, raw, source, mask, final)
                    preview = local_path(task, directory / "working/review.png", exists=False)
                    write_preview(preview, source, mask, raw, final)
                    artifacts.update({"raw_generated": file_record(task, raw), "raw_response": {"path": api_receipt["response_path"], "sha256": api_receipt["response_sha256"]},
                                      "prompt": file_record(task, prompt_path), "request": file_record(task, request_path), "review_preview": file_record(task, preview)})
                    generation = {"provider": "responses_image_generation", "controller": args.controller, "image_model": args.model,
                                  "api_request_id": api_receipt["request_id"], "anchor": {"frame_id": anchor["frame_id"], **file_record(task, anchor["path"])} if anchor else None}
                else:
                    save_png(final, rgb(source))
                    qa = {"outside_mask_changed_pixels": 0, "masked_pixels": 0, "visual_review": "unchanged_source_copy"}
                    generation = {"provider": "none", "reason": "no_lifting_accepted_object_mask_in_frame"}
                artifacts["final"] = file_record(task, final)
                receipt = {"schema_version": "1.0", "recipe_id": recipe_id, "frame_id": job["frame_id"], "source_binding": job["source_binding"],
                           "status": "generated_candidate" if job["masked_pixels"] else "unchanged_source_copy", "evidence": "generated" if job["masked_pixels"] else "observed_copy",
                           "removed_object_ids": job["object_ids"], "parent_masks": job["parent_masks"], "parent_pixels": job["parent_pixels"],
                           "protected_dilation_pixels": job["protected_dilation_pixels"], "camera_policy": "fixed_source_camera",
                           "generation": generation, "qa": qa, "artifacts": artifacts, "promotion_allowed": False, "collision_eligible": False}
                write_json(receipt_path, receipt)
            frame = {"frame_id": job["frame_id"], "image_path": relative(task, final), "mask_path": relative(task, mask),
                     "source_image_path": relative(task, job["source"]), "width": job["width"], "height": job["height"],
                     "image_to_source": np.eye(3).tolist(), "receipt_path": relative(task, receipt_path),
                     "image_sha256": sha256(final), "mask_sha256": sha256(mask), "evidence": receipt["evidence"], "status": receipt["status"]}
            output_frames.append(frame)
            if job["masked_pixels"] and anchor is None:
                anchor = {"frame_id": job["frame_id"], "path": final}
            report["completed_frame_count"] = len(output_frames)
            report["api_requests_reserved_total"] = len(journal["requests"])
            write_json(report_path, report)
        for record in dependencies.values():
            verify_record(task, record)
        for job in jobs:
            verify_record(task, job["source_binding"])
            for record in job["parent_masks"]:
                verify_record(task, record)
    except Exception as error:
        report.update({"status": "blocked", "completed_frame_count": len(output_frames), "api_requests_reserved_total": len(journal["requests"]),
                       "failure_type": type(error).__name__, "failure": str(error) if isinstance(error, CleanPlateError) else "local artifact validation or processing failed", "automatic_retry": False})
        write_json(report_path, report)
        raise
    report.update({"status": "generated_candidates_pending_visual_review", "completed_frame_count": len(output_frames), "api_requests_reserved_total": len(journal["requests"])})
    write_json(report_path, report)
    frames_path = run_dir / "clean_plate_frames.json"
    write_json(frames_path, {"schema_version": "1.0", "evidence": "generated", "camera_policy": "fixed_source_camera", "frames": output_frames,
                             "source_frames_manifest_sha256": dependencies["frames_manifest"]["sha256"], "promotion_allowed": False, "collision_eligible": False})
    publish(args, {"clean_plate_frames": frames_path, "clean_plate_report": report_path})
    return {"status": report["status"], "frames": len(output_frames), "api_requests_reserved_total": len(journal["requests"]), "report_path": relative(task, report_path)}


def run(args, transport=post_response):
    validate_options(args)
    task = args.task_dir.resolve()
    local_path(task, args.outputs, exists=False)
    stage = local_path(task, "stages/clean_plate", exists=False)
    stage.mkdir(parents=True, exist_ok=True)
    with stage_lock(task, stage):
        jobs, dependencies, accepted = load_plan(args)
        return execute_locked(args, task, stage, jobs, dependencies, accepted, transport)


def parser():
    app = provider_parser(__doc__)
    app.add_argument("--endpoint", default="local://lama-inpainting", help="Complete Responses URL, e.g. https://plbbl.com/v1/responses; unused by the local LaMa transport")
    app.add_argument("--token-env", default="PLBBL_API_KEY")
    app.add_argument("--controller", required=True)
    app.add_argument("--model", required=True, help="image_generation tool model, separate from the controller")
    app.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high", "xhigh"), default="high")
    app.add_argument("--quality", choices=("low", "medium", "high", "auto"), default="high")
    app.add_argument("--size", default="source", help="source retains the calibrated width/height; auto or explicit WIDTHxHEIGHT are audited for aspect changes")
    app.add_argument("--dilate-pixels", type=int, default=5)
    app.add_argument("--max-mask-fraction", type=float, default=0.8)
    app.add_argument("--max-ownership-overlap-fraction", type=float, default=0.01,
                     help="Largest share of an accepted mask that may be reassigned to a protected unaccepted object")
    app.add_argument("--max-raw-outside-mae", type=float, default=30.0)
    app.add_argument("--max-aspect-error", type=float, default=0.02)
    app.add_argument("--max-api-requests", type=int, default=0, help="Cumulative task-stage POST budget, including failed or uncertain requests; no implicit retries")
    app.add_argument("--retry-request-id", action="append", default=[], help="Explicitly authorize ONE failed/unknown request ID from the journal to be resubmitted; may incur another charge; the cumulative budget still applies")
    app.add_argument("--timeout-seconds", type=float, default=900)
    app.add_argument("--local-lama-model", type=Path,
                     help="Run offline LaMa inpainting instead of the Responses API; no credential or network is used")
    return app


if __name__ == "__main__":
    try:
        arguments = parser().parse_args()
        transport = local_lama_transport(arguments.local_lama_model) if arguments.local_lama_model else post_response
        print(json.dumps(run(arguments, transport), indent=2))
    except (CleanPlateError, OSError, ValueError, KeyError, ImportError) as error:
        print(f"clean_plate failed: {error if isinstance(error, CleanPlateError) else type(error).__name__}", file=sys.stderr)
        raise SystemExit(2)
