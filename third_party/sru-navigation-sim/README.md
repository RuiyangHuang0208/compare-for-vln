# SRU Navigation Simulation Subset

Only the unmodified B2W Isaac asset required by this workspace is vendored here.

```text
Upstream: https://github.com/leggedrobotics/sru-navigation-sim.git
Commit: da6d5d68e46122967ab82ea9077f8304b5af47d7
USD: isaaclab_nav_task/navigation/assets/data/Robots/B2W/b2w_rsl.usd
USD SHA-256: 5801459641e40b9bd2b48dc76c566f483f5077ffd4097d364fb76fd81a71b0c2
```

The corresponding policy is stored at
`checkpoints/b2w_locomotion/isaac_pt/policy_b2w_new_2.pt`. Runtime adaptation stays in
`src/robot_controller/scripts/rsl_rl/play.py`; the upstream asset is not modified.
