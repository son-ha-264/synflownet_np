from dataclasses import dataclass, field
from typing import List, Optional

from synflownet.utils.misc import StrictDataClass


@dataclass
class SEHTaskConfig(StrictDataClass):
    reduced_frag: bool = False


@dataclass
class ReactionTaskConfig:
    templates_filename: str = "hb.txt"
    reverse_templates_filename: Optional[str] = None
    reward: Optional[str] = None
    building_blocks_filename: str = "enamine_bbs.txt"
    precomputed_bb_masks_filename: str = "precomputed_bb_masks_enamine_bbs.pkl"
    building_blocks_costs: Optional[str] = None
    sanitize_building_blocks: bool = False
    ### NEW: ---- ReactionT5 forward model as extra "virtual" bimolecular reactions ----
    use_reaction_t5: bool = False
    t5_model_name: str = "/home/sonh/MAINCE_project/pretrained/rt5_ft_uspto_npl_custom_replay01_kl01"
    t5_num_bi_slots: int = 3  # K extra children per (state, bb) pair via bimolecular ReactionT5 (0 disables)
    t5_num_uni_slots: int = 5  # K extra children per state via unimolecular ReactionT5, run on the state alone (0 disables)
    t5_num_beams: int = 10  # beams searched to find K distinct, valid products
    t5_device: str = "cuda"
    t5_bb_mask: str = "template_reactive"  # "all" | "template_reactive": which BBs can be the 2nd reactant of a T5 reaction
    t5_max_heavy_atoms: Optional[int] = 100  # discard T5 products larger than this
    t5_max_state_heavy_atoms: Optional[int] = None  # forbid T5 actions on states larger than this
    t5_provenance_db: Optional[str] = None  # SQLite path to share recorded T5 edges across workers


@dataclass
class VinaConfig(StrictDataClass):
    opencl_binary_path: str = (
        "bin/QuickVina2-GPU-2-1"  # needed if you use VINA for rewards
    )
    vina_path: str = (
        "bin/QuickVina2-GPU-2-1/Vina-GPU"  # path to VINA executable, needed if you use VINA for rewards
    )
    target: str = "kras"  # kras, 2bm2


@dataclass
class TasksConfig(StrictDataClass):
    reactions_task: ReactionTaskConfig = field(default_factory=ReactionTaskConfig)
