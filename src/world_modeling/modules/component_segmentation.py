from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "component_segmentation", "SAM3/SAM3-I component segmentation",
    ("frames_manifest", "object_proposals", "object_descriptions"),
    (role("component_mask_candidates", "derived", "application/json"), role("physical_instance_hypotheses", "derived", "application/json")),
    "sam3_i", "Frame-local instance candidates and parent identity hypotheses; geometry validation belongs to lifting."
)
