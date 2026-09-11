from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "geometry_completion", "Stream3D and 3D object completion",
    ("object_orbit_videos", "assembled_object_views", "isolated_object_ply", "object_descriptions", "cameras", "scene_gaussian_ply", "lifting_report"),
    (role("completion_candidates", "generated", "application/json"), role("completed_object_meshes", "generated", "model/gltf-binary")),
    "stream3d_trellis_sam3d_hunyuan", "Routes Stream3D plus TRELLIS/TRELLIS2/SAM3D/Hunyuan3D 2.1 through a single candidate contract."
)
