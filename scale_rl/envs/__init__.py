from scale_rl.envs.base import ScaleRLBase
from scale_rl.envs.dapo_env import DapoMathEnv
from scale_rl.envs.livecodebench import LiveCodeBenchEnv

DATASET_ENV_CLS: dict[str, type[ScaleRLBase]] = {
    "dapo": DapoMathEnv,
    "livecodebench": LiveCodeBenchEnv,
}
