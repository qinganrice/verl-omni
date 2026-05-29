import verl_omni  # noqa: F401 — triggers model/rollout registration

# Eagerly import sub-modules before ray.init() so their side effects
# (env vars, CUDA state) are present when Ray spawns worker processes.
import verl_omni.pipelines  # noqa: F401
import verl_omni.reward_loop  # noqa: F401
import verl_omni.workers.engine  # noqa: F401
import verl_omni.workers.rollout  # noqa: F401

from verl.trainer.main_ppo import main  # noqa: E402

if __name__ == "__main__":
    main()
