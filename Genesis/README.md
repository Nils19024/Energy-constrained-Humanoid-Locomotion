# Genesis

All attempts to train robots using Genesis failed. Genesis is still in early stages and can thus be very sensitive to versioning. As an example, as of the recent update 0.3.0, robots fall through the ground, behaviour not observed before.
The G1 robot is not natively supported at this point, it thus has to be imported from MJCF .xml or a .urdf file, some configurations have to be done manually.

## Important files overview
```g1_train.py:``` \
Script to train an agent using the PPO algorithm.

```g1_eval.py:``` \
Creates a video of the trained agent.

```g1_env.py:``` \
These files were adapted to the G1 robot.

## Installation

Follow the steps on https://genesis-world.readthedocs.io/en/latest/#quick-installation.