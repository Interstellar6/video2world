from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "mesh_postprocess", "Mesh repair, simplify, UV bake, and CoACD",
    ("completion_candidates", "completed_object_meshes"),
    (role("repaired_visual_meshes", "derived", "model/gltf-binary"), role("uv_textures", "derived", "image/png"), role("coacd_collision_meshes", "derived", "model/obj", collision_eligible=True), role("mesh_qa_report", "derived", "application/json")),
    "embodiedgen_v2_geometry_tools", "3D-Fixer is a completion model; this stage runs mesh utilities and verified convex decomposition."
)
