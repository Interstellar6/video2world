from .background_reconstruction import SPEC as BACKGROUND_RECONSTRUCTION
from .clean_plate import SPEC as CLEAN_PLATE
from .component_assembly import SPEC as COMPONENT_ASSEMBLY
from .component_segmentation import SPEC as COMPONENT_SEGMENTATION
from .geometry_completion import SPEC as GEOMETRY_COMPLETION
from .mesh_postprocess import SPEC as MESH_POSTPROCESS
from .object_lifting import SPEC as OBJECT_LIFTING
from .orbit_video import SPEC as ORBIT_VIDEO
from .physics_estimation import SPEC as PHYSICS_ESTIMATION
from .scene_recomposition import SPEC as SCENE_RECOMPOSITION
from .scene_reconstruction import SPEC as SCENE_RECONSTRUCTION
from .scene_understanding import SPEC as SCENE_UNDERSTANDING

ORDERED_SPECS = (
    SCENE_RECONSTRUCTION,
    SCENE_UNDERSTANDING,
    COMPONENT_SEGMENTATION,
    OBJECT_LIFTING,
    CLEAN_PLATE,
    BACKGROUND_RECONSTRUCTION,
    COMPONENT_ASSEMBLY,
    ORBIT_VIDEO,
    GEOMETRY_COMPLETION,
    MESH_POSTPROCESS,
    PHYSICS_ESTIMATION,
    SCENE_RECOMPOSITION,
)
from ..registry import ModuleRegistry

REGISTRY = ModuleRegistry(ORDERED_SPECS)
REGISTRY.load_plugins()
ORDERED_SPECS = REGISTRY.modules
SPECS = REGISTRY.specs
