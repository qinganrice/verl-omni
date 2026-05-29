import verl_omni  # noqa: F401 — triggers model/rollout registration

# Import rollout worker in the main process so vllm's os.environ side effects
# (NCCL env vars) are set before ray.init() and inherited by Ray worker processes.
import verl_omni.workers.rollout  # noqa: F401

from verl.trainer.main_ppo import main  # noqa: E402

if __name__ == "__main__":
    main()
