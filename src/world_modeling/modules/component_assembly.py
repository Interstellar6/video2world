from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "component_assembly", "Extract observed views of verified object composites",
    ("object_descriptions", "component_masks", "isolated_object_ply", "lifting_report", "cameras"),
    (role("assembled_object_views", "derived", "application/json"), role("assembly_report", "derived", "application/json")),
    "component_assembler", "Extracts same-camera verified parent RGBA views; component annotations do not alter alpha or complete hidden surfaces."
)
