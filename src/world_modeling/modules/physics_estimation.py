from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "physics_estimation", "Qwen-VL physical-property estimation and OBJ asset",
    ("object_descriptions", "repaired_visual_meshes", "coacd_collision_meshes", "mesh_qa_report"),
    (role("physical_object_obj", "derived", "model/obj", collision_eligible=True), role("physics_properties", "derived", "application/json"), role("physics_report", "derived", "application/json")),
    "embodiedgen_v2_qwen_vl", "Estimates scale, mass, and smoothness; every estimate must carry units, confidence, and source evidence."
)

