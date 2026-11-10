# Observation and information-path audit

Read-only source and counterfactual checks on November 9, 2026. No experiment
code, configuration, checkpoint or metric was changed.

`GridEnv._views()` constructs a square egocentric crop. For the base `types`
task (v1 and vocal), it does not compute wall occlusion. A scout close to the
partition can therefore see warehouse cells, including item features, within
the crop. The wall physically prevents it from reaching those items. For
`compositional` (v2), a separate branch masks the scout's item/depot channels
and warehouse wall features.

Neither view contains other-agent position features. `GridEnv.observe()` gives
each agent its own position, carried-item block, masked target cue, received
message and time. `cue_mask()` hides the target from the runner in comm,
no_comm and random_msg. It reveals the target in the oracle reference, whose
observation space has the same size but a different cue mask.

A constructed base-task counterfactual confirmed that changing only the
target leaves the uncued runner observation identical while changing the
scout observation. Changing the scout position also leaves the runner
observation identical. The same construction exposes warehouse content to
the v1 scout and confirms that the v2 scout content channels are masked.
The precise receipt is preserved in the ignored course audit directory.

These checks support the base task's target-information asymmetry. They do
not claim photorealistic visibility or prove that the wall itself implements
visual privacy. The report and narration describe the actual crop and masks.
The v2 unseen target-support shortcut is a separate limitation already
disclosed in the report; a uniform guessing bound is not applied to that split.

## Training-objective cross-check

For the compositional task, `train.py` supplies `n_values` to `ppo_update`.
`joint_cue` converts the color and shape blocks to a joint-cue one-hot;
`word_cue_information` sums each message position's information about that
joint cue. The loss does not assign color to word 1 or shape to word 2, and
does not penalize redundant encoding. The report, guide and narration state
this explicitly. The frozen implementation remains unchanged.
