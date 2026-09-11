from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "scene_understanding", "Qwen-VL object discovery and structured descriptions",
    ("frames_manifest",),
    (role("object_proposals", "derived", "application/json"), role("object_descriptions", "derived", "application/json")),
    "qwen_vl_2_5", "Requires frame-specific boxes, detailed component-aware descriptions, prompt version, and confidence."
)
