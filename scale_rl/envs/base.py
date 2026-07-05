import abc
from typing import Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

from scale_rl.inference.rollout_worker import vLLMRollout


class ScaleRLBase(BaseTextEnv):
    """
    Base scale_rl environment for scale_rl datasets.

    Subclasses must implement:
      - init:           build opening conversation from prompt (dataset-specific)
      - compute_reward: per-step correctness signal for the RL reward
      - evaluate:       dataset-level benchmark eval (e.g. AIME pass@k), logs to wandb
    """

    def __init__(self, kind: str, dataset: str) -> None:
        super().__init__()
        self.kind = kind
        self.dataset = dataset
        self.high_pass_rate: bool = False

    @abc.abstractmethod
    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        """Build the opening conversation. Dataset-specific."""
        ...

    def step(self, action: str) -> BaseTextEnvStepOutput:
        reward, done = self.compute_reward(action)
        self.turns += 1
        return BaseTextEnvStepOutput(
            observations=[{"role": "assistant", "content": action}],
            reward=reward,
            done=done or self.turns >= self.max_turns,
            metadata={"kind": self.kind, "dataset": self.dataset},
            postprocessed_action=None,
        )

    @abc.abstractmethod
    def compute_reward(self, action: str) -> tuple[float, bool]:
        """Return (reward, done) for a single model completion."""
        ...

    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        rollout_worker: vLLMRollout,
        step: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run dataset-level evaluation (e.g. AIME pass@k, LCB pass@1) using the
        already weight-synced rollout worker. Log results to wandb and return a
        metrics dict keyed like {"eval/pass@8": 0.42}.
        """
        ...