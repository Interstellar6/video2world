from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "object_lifting", "Depth-gated 2D-to-3D lifting and scene carve",
    ("cameras", "scene_depth", "scene_gaussian_ply", "scene_tsdf_mesh", "component_mask_candidates", "physical_instance_hypotheses"),
    (role("isolated_object_ply", "derived", "application/octet-stream"), role("carved_scene_ply", "derived", "application/octet-stream"), role("lifting_report", "derived", "application/json"),
     role("component_masks", "derived", "application/json"), role("physical_instance_tracks", "derived", "application/json")),
    "depth_anything_3", "Only accepted cross-view physical tracks may carve/lift; report holes and coverage explicitly."
)
