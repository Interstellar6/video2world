from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "scene_recomposition", "Reinsert independently modeled objects into scene",
    ("scene_gaussian_ply", "carved_scene_ply", "generated_background_gaussian_ply", "repaired_visual_meshes", "physical_object_obj", "physics_properties", "cameras", "lifting_report", "isolated_object_ply"),
    (role("world_manifest", "derived", "application/json"), role("visual_scene_manifest", "derived", "application/json"), role("collision_scene_manifest", "derived", "application/json"), role("recomposition_report", "derived", "application/json")),
    "world_assembler", "Visual scene may layer PGSR and generated candidates; collision scene may reference only validated physical OBJ/CoACD assets."
)
