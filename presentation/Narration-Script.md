# Narration script

Use this as a draft in your own voice. Edit it so every statement reflects your understanding. Aim for 7–9 minutes; no fixed video duration was stated in the assignment. Advance slides manually while recording.


## 1. Can agents invent useful messages?

My project asks whether two independently learning agents can invent a useful message protocol when I never assign meanings to their symbols. I built a small world where one agent has the information and the other can act on it. I will show what changed in the learning method, the controlled results, and an extension where communication works but the planned generalization claim fails.


## 2. The scout knows. The runner acts.

The scout is in the control room and knows which item is wanted. The runner is in the warehouse, sees only a local window, and does not know the target. Neither agent's visual input contains the other agent; the wall prevents physical contact. The local view is a square crop, rather than line-of-sight vision. Geometry is sampled before the target, and every item must be deliverable, so the map does not leak the answer. Messages arrive one step after they are sent. A wrong delivery ends the episode.


## 3. One targeted change to independent PPO

The starting method is independent recurrent PPO. Each agent has its own recurrent policy, value function and optimizer. The modification adds a cue-information objective for agents that actually see a cue. It encourages different message distributions for different targets without deciding which symbol should mean which target. This adapts the positive-signalling idea of Eccles and colleagues. I am not claiming to have invented that idea. The direct ablation removes only this objective, and is labeled exploratory because it was added after the original results were known.


## 4. Does the added objective help?

This direct comparison removes only the cue-information objective. The architecture, environment, optimizer and full training budget are equal, with twenty seeds per method. Unbiased IPPO reaches 22.7% held-out success, compared with 87.6% for cue-directed IPPO. The difference is 64.9 percentage points, with a ninety-five-percent seed-bootstrap interval from +63.3 to +66.4 pp. The interval favors the cue-directed method in this selected task and configuration. This is exploratory because it was added after the original outcomes were known. It does not compare against the complete Eccles method or establish a new state of the art.


## 5. A benchmark of trained policies

The main comparison has learned communication, no communication, random messages, and a full-information oracle. The observation and action spaces, network and budget are matched within each experiment. Every condition has twenty independent training seeds. Each policy is evaluated on two hundred fixed episodes per split. Confidence intervals bootstrap the training seeds, rather than treating all episodes as independent trained systems. Pilots use separate seeds and training layouts, and the benchmark evaluates only final checkpoints.


## 6. Communication improves held-out success

On held-out layouts, learned communication succeeds 87.6% of the time, compared with 22.5% without communication. The difference is 65.0 percentage points. Random messages give little benefit, while the oracle indicates how well the runner can navigate when it already knows the target. These are aggregate results across the twenty seeds. The scripted lines are privileged references, not learned competitors.


## 7. Does the listener use the meanings?

Here is the same held-out episode under intact and permuted messages. The scout and runner use the same learned networks, but a fresh permutation changes the received symbol identities. Watch the runner's target choice. Across all seeds, the intact-minus-permuted drop is 67.6 percentage points. This supports reliance on stable message meanings. A replay alone would not establish that result; the controlled aggregate comparison does.


## 8. Informative speech is not enough

The representative codebook contains arbitrary symbol assignments and some synonyms. But seeing a neat codebook does not prove communication. The signalling objective also makes scouts informative in conditions where nobody hears their outputs. The mutual-information statistic describes the speaker; it does not prove the runner listens. This follows the measurement concern raised by Lowe and colleagues and is why the intervention test matters.


## 9. Two words do not guarantee generalization

The second experiment asks about targets with a color and a shape, communicated using two words. The objective rewards each word's information about the joint cue; it does not enforce one word per property. Five combinations never appear in training, even as distractors. Communication reaches 37.1% on seen combinations and 26.7% on unseen combinations. The frozen communication claim passes, but the generalization claim fails its predefined effect-size requirement. The later claim is therefore not reached. I retain this negative result rather than tuning after seeing it.


## 10. Read the unseen result carefully

The unseen task has a limitation that matters for interpretation. Its target combinations form a fixed one-to-one matching. A strategy that knows this support can often identify the target without learning a compositional message. The source-only support calculation is shown here; it is not learned navigation performance. The fixed fifty-percent reference was preserved, but it cannot be called a one-property ceiling for this split. The lexicon and structural metrics are descriptive, not proof of compositional language.


## 11. Exploratory: a continuous vocal channel

An exploratory extension replaces discrete symbols with a noisy continuous pitch. The agents develop distinct pitch categories, and hearing a voice from another episode removes much of the benefit. This tests whether the listener uses the actual signal rather than just its presence. The vocal arm has a longer training budget, so its absolute success rate is not a clean one-variable comparison with the discrete arm. It remains outside the original confirmatory hypotheses.


## 12. The implementation runs from saved weights

For the live demonstration I run fresh inference from the included final checkpoints. This command loads the learned policies and computes actions again; it does not merely play a saved animation. It writes a new output directory containing episode records, metrics, an execution receipt and regenerated replays. During preparation, all fourteen representative-seed summary rows matched the canonical results exactly. This is a representative execution check, not a rerun of the entire training sweep. I can run this command in the terminal during the live presentation.


## 13. Useful coordination, with bounded claims

The main result is that a task-specific signalling bias can support useful coordination between independently learning agents in this world, with interventions showing that the runner uses stable message identities. The extension shows the limit: communication success does not automatically give compositional generalization. The code, data and report preserve the controls, negative findings and exploratory labels. The work builds on PPO, independent multi-agent learning and prior emergent-communication methods. Claude and ChatGPT or Codex assisted with planning, implementation, checks and report refinement, as disclosed in the report. This recording uses my own narration.
