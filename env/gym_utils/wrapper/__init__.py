from .multi_step import MultiStep
from .robomimic_lowdim import RobomimicLowdimWrapper
from .robomimic_image import RobomimicImageWrapper
from .d3il_lowdim import D3ilLowdimWrapper
from .mujoco_locomotion_lowdim import MujocoLocomotionLowdimWrapper
from .sparse_reward import SparseRewardWrapper, BinarySuccessWrapper, ThresholdRewardWrapper


wrapper_dict = {
    "multi_step": MultiStep,
    "robomimic_lowdim": RobomimicLowdimWrapper,
    "robomimic_image": RobomimicImageWrapper,
    "d3il_lowdim": D3ilLowdimWrapper,
    "mujoco_locomotion_lowdim": MujocoLocomotionLowdimWrapper,
    "sparse_reward": SparseRewardWrapper,
    "binary_success": BinarySuccessWrapper,
    "threshold_reward": ThresholdRewardWrapper,
}
