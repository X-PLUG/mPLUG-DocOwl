import os
from tinychart.model.multimodal_encoder.siglip_encoder import SigLipVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    return SigLipVisionTower(vision_tower, vision_tower_cfg, **kwargs)