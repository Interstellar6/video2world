from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "scene_reconstruction", "Holi-Spatial scene reconstruction",
    ("source_media",),
    (role("frames_manifest", "observed", "application/json"), role("cameras", "observed", "application/json"), role("scene_depth", "observed", "application/json"), role("scene_gaussian_ply", "observed", "application/octet-stream"), role("scene_tsdf_mesh", "observed", "model/gltf-binary")),
    "holi_spatial", "Holi-Spatial owns DA3/VGGT-Omega depth, PGSR visual PLY, and TSDF mesh."
)

