from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "clean_plate", "Multi-view clean-plate image repair",
    ("frames_manifest", "component_masks", "physical_instance_tracks", "lifting_report"),
    (role("clean_plate_frames", "generated", "application/json"), role("clean_plate_report", "generated", "application/json")),
    "image_generation", "Generated pixels are a reconstruction hypothesis, never observed geometry or collision evidence."
)

