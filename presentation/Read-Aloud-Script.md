# Word-for-word presentation script

Read the paragraphs aloud. Bracketed directions are actions, not spoken words.
This follows the thirteen-slide deck. Allow about nine minutes for speech,
plus any live execution. Read percentages as “percent” and PPO as “P–P–O.”
Review the wording and make sure you can explain each claim before recording.

The local slide recorder captures the slides and your voice. It does not
capture switching to a terminal. For the live Zoom presentation, use the
separate live-demo insert at the end, immediately after slide 6, then return
to slide 7. Slide 12 can be brief because you have already shown execution.

## 1. Can agents invent useful messages?

Hi, I'm Christian Tadros, and my project is Emergent Communication in a Grid World.

The question is whether two independently learning agents can develop a useful communication protocol when nobody programs the meanings of their messages.

This matters because cooperating agents often have different information. Reliable coordination can depend on information held by one agent reaching the agent that can use it.

I built a controlled reinforcement-learning task to study that problem. The project includes the environment, recurrent learning agents, controlled experiments, and tests of whether the listener actually uses the messages.

[Advance to slide 2.]

## 2. The scout knows. The runner acts.

There are two agents: a scout and a runner.

The scout knows which item is wanted, but cannot reach the items. The runner can collect and deliver items, but does not know which one is the target.

Their observations omit the other agent's position, so the scout cannot communicate by moving around. Each agent sends a symbol, and its partner receives it one step later.

The runner has a limited local view. Delivering the wrong item ends the episode, so it cannot simply try every item until it gets the right one.

[Advance to slide 3.]

## 3. One targeted change to independent PPO

The starting method is independent recurrent PPO, or Proximal Policy Optimization. Each agent has its own neural network, memory, value estimate, and optimizer.

My modification adds an auxiliary objective that encourages the scout's messages to contain information about its private target cue.

It groups predicted distributions by the private cue and encourages different cue groups to have different message distributions. It never specifies which symbol should represent which target, and it supplies no supervised decoder to the listener.

This adapts the positive-signalling idea of Eccles and colleagues. The contribution is the task-specific adaptation and controlled evaluation. PPO and positive signalling are established prior work. This comparison does not establish superiority to the full Eccles method. Local observations, recurrent memory, separate policies and shared rewards implement the course's agent and multi-agent concepts.

[Advance to slide 4.]

## 4. Does the added objective help?

To isolate that modification, I compared the cue-directed method with unbiased IPPO, which still has a message channel but has the extra objective removed.

The architecture, environment, optimizer, and full training budget are matched, with twenty training seeds per method.

Unbiased IPPO reaches 22.7 percent held-out success. The cue-directed method reaches 87.6 percent. That is a difference of 64.9 percentage points, with a ninety-five-percent confidence interval from 63.3 to 66.4 points.

This comparison is explicitly exploratory because it was added after the original results were known. Its conclusion is limited to this task and selected configuration.

[Advance to slide 5.]

## 5. A benchmark of trained policies

The main benchmark has four conditions: learned communication, no communication, random messages, and an oracle that reveals the target to the runner.

There are twenty independently trained pairs per condition, giving eighty runs in the main v1 experiment. Each run uses about ten million environment steps.

Each final policy is evaluated on two hundred fixed episodes per split. The same test episodes are reused across policies for comparison; they are not thousands of distinct test scenarios.

Success means delivering the correct item within the time limit. Confidence intervals resample training seeds, because the independently trained pair is the unit of replication. Evaluation uses final checkpoints, and the small selection pilots use separate seeds and training layouts.

[Advance to slide 6.]

## 6. Communication improves held-out success

On held-out layouts, learned communication reaches 87.6 percent success, compared with 22.5 percent without communication. The difference is 65.0 percentage points.

Random messages provide little benefit. The oracle gives a learned navigation reference when the runner already knows the target. The scripted lines use privileged map knowledge and are labeled as references.

This comparison shows the value of receiving useful information. The earlier unbiased-IPPO comparison separately tests the added training objective.

[For the live Zoom presentation, perform the live-demo insert below now. For a slide-only recording, advance directly to slide 7.]

## 7. Does the listener use the meanings?

Here is the same held-out episode with the same trained networks under two message conditions.

On the left, the messages are intact. On the right, the symbol identities are scrambled before the runner hears them. Watch how the outcome changes.

[Let the paired animation play through.]

Across all twenty seeds, scrambling reduces success by 67.6 percentage points. That supports the conclusion that the listener depends on stable message identities. This one replay illustrates the aggregate result; it does not establish the result by itself.

[Advance to slide 8.]

## 8. Informative speech is not enough

This plot shows a representative learned codebook. The symbol assignments are arbitrary, and some targets have more than one symbol associated with them.

However, a neat codebook alone does not prove communication. The auxiliary objective can make scouts informative even when their messages are never delivered.

Mutual information describes the speaker. The message intervention tests whether the listener depends on the channel. This distinction follows the measurement concern raised by Lowe and colleagues.

[Advance to slide 9.]

## 9. Two words do not guarantee generalization

V2 asks a harder question: can a two-word protocol generalize to color-and-shape combinations withheld from training?

Five combinations are absent from training, including as distractors. The objective rewards each word's information about the joint cue, but does not enforce one word for color and one for shape.

Communication reaches 37.1 percent success on seen combinations and 26.7 percent on unseen combinations.

The pre-registered communication claim is supported, but the generalization claim fails its predefined minimum-effect requirement. I retain that negative result. The seen rule requires a point gain of at least fifteen percentage points and an interval excluding zero, not a lower confidence bound of fifteen points.

A retrospective audit of the saved evaluations finds wrong-item picks in 47.8 percent of seen episodes and 57.0 percent of unseen episodes. The oracle also declines on unseen cases. These descriptions do not isolate a language failure from navigation or unfamiliar task combinations. Useful communication here does not establish compositional generalization.

[Advance to slide 10.]

## 10. Read the unseen result carefully

There is also a limitation in the unseen benchmark. The held-out target combinations form a fixed one-to-one matching.

A strategy that knows which combinations belong to that set can sometimes identify the target from the available items, without a compositional message.

The 87.5 percent calculation shown here is theoretical identification before navigation. It is not a learned agent's success rate.

For that reason, the fixed fifty-percent reference cannot be treated as proof that both properties transferred. The structural metrics and codebook also remain descriptive evidence.

[Advance to slide 11.]

## 11. Exploratory: a continuous vocal channel

I also explored a continuous channel, where the agents emit noisy pitches instead of discrete symbols.

The agents develop separated pitch categories. Replacing the voice with one from another episode removes much of the benefit, supporting dependence on the actual signal.

This extension is exploratory and uses a longer training budget. Its absolute success rate therefore does not isolate the effect of changing the channel type.

[Advance to slide 12.]

## 12. The implementation runs from saved weights

The code package includes final trained models and a command for fresh execution.

That command loads the policies, computes actions again, and writes new episode records, metrics, an execution receipt, and regenerated replays.

The aggregate scientific claims come from the full benchmark. The live command demonstrates execution of representative trained policies; it does not repeat the many-hour training sweep.

This makes the demonstration reproducible from the submitted package, with the software environment installed beforehand.

[Advance to slide 13.]

## 13. Useful coordination, with bounded claims

The main finding is that a cue-directed signalling objective supports useful coordination in this split-information task, and the message interventions show that the runner depends on stable message identities.

The extension establishes an important limit: communication success does not automatically produce compositional generalization.

The project connects to the course's agent reinforcement learning and multi-agent systems material. Its contribution is a controlled implementation and evaluation, with the comparisons, negative findings, and limitations reported explicitly.

Claude and ChatGPT or Codex assisted with planning, implementation, testing, and report refinement, as disclosed in the report. I remain responsible for understanding and explaining the work. For the final recording I will provide my own narration; the separately labeled review copy uses temporary local synthetic speech.

Thank you.

[For a recording: Stop and save, then listen to the complete video.]

## Live-demo insert — after slide 6, during the Zoom presentation

[Do not read these directions aloud. Share your screen, switch to a terminal in the extracted project folder, and run the command below. Install the environment and rehearse it before the call. The slide-only recorder does not capture this terminal.]

```bash
uv run --offline --frozen python scripts/demo_inference.py --phase both
```

I will now run fresh inference from the submitted final checkpoints. This loads the learned policies and computes their actions again. It creates a new output folder and preserves the original benchmark results.

[Wait for successful completion. Open the newly printed output directory and its viewer/index.html. If execution fails, describe the failure honestly and identify any saved output as a backup.]

The command has finished. These printed rates describe the representative seeds used for this demonstration. The report's headline results average across all twenty seeds.

Here is a replay generated by this execution. First I will play it with intact messages, and then switch to permuted messages for the same episode.

[Play the episode under Comm: intact. Switch to Comm: permuted and play again.]

The task, target, and learned networks stay the same. The change is the received symbol identities. This makes the intervention visible; the all-seed comparison establishes the size of its effect.

[Return to slide 7. Keep the remaining slides brief and finish by approximately minute eleven, leaving time for questions before the hard fifteen-minute cutoff. Join Zoom only at your assigned slot.]
