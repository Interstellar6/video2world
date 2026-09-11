from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "background_reconstruction", "Reconstruct background from clean plates",
    ("clean_plate_frames", "clean_plate_report", "cameras"),
    (role("generated_background_gaussian_ply", "generated", "application/octet-stream"), role("generated_background_mesh", "generated", "model/gltf-binary"), role("background_reconstruction_report", "generated", "application/json")),
    "holi_spatial", "The reconstructed background remains generated/candidate until separate visual, geometry, and collision QA accept it."
)

