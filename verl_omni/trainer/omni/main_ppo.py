import verl_omni

# Eagerly import sub-modules before ray.init() so their side effects
# (env vars, CUDA state) are present when Ray spawns worker processes.
import verl_omni.pipelines  # noqa: F401
import verl_omni.reward_loop  # noqa: F401
import verl_omni.workers.engine  # noqa: F401
import verl_omni.workers.rollout  # noqa: F401

from verl.trainer.main_ppo import main  # noqa: E402

# Register verl-omni's adapters (rollout, model arch, processor) with upstream
# verl before launching. Ray workers re-run this via verl_omni._init_worker.
verl_omni.bootstrap()

if __name__ == "__main__":
    main()
