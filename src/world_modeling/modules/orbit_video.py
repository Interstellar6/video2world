from ..contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "orbit_video", "FixAnything object orbit-video generation",
    ("assembled_object_views", "assembly_report"),
    (role("object_orbit_videos", "generated", "application/json"), role("orbit_video_report", "generated", "application/json")),
    "fix_anything", "Produces three 360-degree object orbits; outputs are conditioned synthesis, not source observation."
)

