"""verl-omni entry point: import triggers replica registration, then delegate."""

import verl_omni

verl_omni.bootstrap()

from verl.trainer.main_ppo import main  # noqa: E402

if __name__ == "__main__":
    main()
