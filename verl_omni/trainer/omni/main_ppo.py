"""verl-omni entry point: import triggers replica registration, then delegate."""

import verl_omni  # noqa: F401
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
