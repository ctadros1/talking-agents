# Student video narration script

Record only the paragraphs. Do not read the numbered headings. Pause for about two seconds between sections. Read PPO as “P–P–O,” v1 as “version one,” v2 as “version two,” and API as “A–P–I.” Record at a comfortable pace; the video will be adjusted to your narration. You do not need to operate the simulator or match the temporary video's timing while recording. If you stumble, pause and repeat the sentence.

## 1. Can agents invent useful messages?

Hi, I'm Christian Tadros, and my project is Emergent Communication in a Grid World.

The question is whether two independently learning agents can develop a useful communication protocol when nobody programs the meanings of their messages.

This matters because cooperating agents often have different information. Reliable coordination can depend on information held by one agent reaching the agent that can use it.

I built a controlled reinforcement-learning task to study that problem. The project includes the environment, recurrent learning agents, controlled experiments, and tests of whether the listener actually uses the messages.

## 2. The scout knows. The runner acts.

There are two agents: a scout and a runner.

The scout knows which item is wanted, but cannot reach the items. The runner can collect and deliver items, but does not know which one is the target.

Their observations omit the other agent's position, so the scout cannot communicate by moving around. Each agent sends a symbol, and its partner receives it one step later.

The runner has a limited local view. Delivering the wrong item ends the episode, so it cannot simply try every item until it gets the right one.

## 3. One targeted change to independent PPO

The starting method is independent recurrent PPO, or Proximal Policy Optimization. Each agent has its own neural network, memory, value estimate, and optimizer.

My modification adds an auxiliary objective that encourages the scout's messages to contain information about its private target cue.

It groups predicted distributions by the private cue and encourages different cue groups to have different message distributions. It never specifies which symbol should represent which target, and it supplies no supervised decoder to the listener.

This adapts the positive-signalling idea of Eccles and colleagues. The contribution is the task-specific adaptation and controlled evaluation. PPO and positive signalling are established prior work. This comparison does not establish superiority to the full Eccles method. Local observations, recurrent memory, separate policies and shared rewards implement the course's agent and multi-agent concepts.

## 4. Does the added objective help?

To isolate that modification, I compared the cue-directed method with unbiased independent PPO, which still has a message channel but has the extra objective removed.

The architecture, environment, optimizer, and full training budget are matched, with twenty training seeds per method.

Unbiased independent PPO reaches 22.7 percent held-out success. The cue-directed method reaches 87.6 percent. That is a difference of 64.9 percentage points, with a ninety-five-percent confidence interval from 63.3 to 66.4 points.

This comparison is explicitly exploratory because it was added after the original results were known. Its conclusion is limited to this task and selected configuration.

## 5. A benchmark of trained policies

The main benchmark has four conditions: learned communication, no communication, random messages, and an oracle that reveals the target to the runner.

There are twenty independently trained pairs per condition, giving eighty runs in the main v1 experiment. Each run uses about ten million environment steps.

Each final policy is evaluated on two hundred fixed episodes per split. The same test episodes are reused across policies for comparison; they are not thousands of distinct test scenarios.

Success means delivering the correct item within the time limit. Confidence intervals resample training seeds, because the independently trained pair is the unit of replication. These intervals describe training variability on the fixed evaluation set, rather than uncertainty across arbitrary task designs. Evaluation uses final checkpoints, and the small selection pilots use separate seeds and training layouts.

## 6. Communication improves held-out success

On held-out layouts, learned communication reaches 87.6 percent success, compared with 22.5 percent without communication. The difference is 65.0 percentage points.

Random messages provide little benefit. The oracle gives a learned navigation reference when the runner already knows the target. The scripted lines use privileged map knowledge and are labeled as references.

This comparison shows the value of receiving useful information. The earlier unbiased-independent PPO comparison separately tests the added training objective.

## 7. Does the listener use the meanings?

Here is recorded, fresh browser inference for the same held-out scenario, using the same trained networks under two message conditions.

On the left, the messages are intact. On the right, the symbol identities are scrambled before the runner hears them. Watch how the outcome changes.

Across all twenty seeds, scrambling reduces success by 67.6 percentage points. That supports the conclusion that the listener depends on stable message identities. This representative execution illustrates the aggregate result; it does not establish the result by itself.

## 8. Informative speech is not enough

This plot shows a representative learned codebook. The symbol assignments are arbitrary, and some targets have more than one symbol associated with them.

However, a neat codebook alone does not prove communication. The auxiliary objective can make scouts informative even when their messages are never delivered.

Mutual information describes the speaker. The message intervention tests whether the listener depends on the channel. This distinction follows the measurement concern raised by Lowe and colleagues.

## 9. Two words do not guarantee generalization

V2 asks a harder question: can a two-word protocol generalize to color-and-shape combinations withheld from training?

Five combinations are absent from training, including as distractors. The objective rewards each word's information about the joint cue, but does not enforce one word for color and one for shape.

Communication reaches 37.1 percent success on seen combinations and 26.7 percent on unseen combinations.

The pre-registered communication claim is supported, but the generalization claim fails its predefined minimum-effect requirement. I retain that negative result. The seen rule requires a point gain of at least fifteen percentage points and an interval excluding zero, not a lower confidence bound of fifteen points.

A retrospective audit of the saved evaluations finds wrong-item picks in 47.8 percent of seen episodes and 57.0 percent of unseen episodes. The oracle also declines on unseen cases. These descriptions do not isolate a language failure from navigation or unfamiliar task combinations. Useful communication here does not establish compositional generalization.

## 10. Read the unseen result carefully

There is also a limitation in the unseen benchmark. The held-out target combinations form a fixed one-to-one matching.

A strategy that knows which combinations belong to that set can sometimes identify the target from the available items, without a compositional message.

The 87.5 percent calculation shown here is theoretical identification before navigation. It is not a learned agent's success rate.

For that reason, the fixed fifty-percent reference cannot be treated as proof that both properties transferred. The structural metrics and codebook also remain descriptive evidence.

## 11. Exploratory: a continuous vocal channel

I also explored a continuous channel, where the agents emit noisy pitches instead of discrete symbols.

The agents develop separated pitch categories. Replacing the voice with one from another episode removes much of the benefit, supporting dependence on the actual signal.

This extension is exploratory and uses a longer training budget. Its absolute success rate therefore does not isolate the effect of changing the channel type.

## 12. The implementation runs from saved weights

Here, the Python command launches a fresh inference run. Watch the process identifier, elapsed timer, new output, and successful exit.

It loads the included final policies, computes actions again, and writes fresh episode records, metrics, an execution receipt, and regenerated replays. The v1 demonstration uses training seed nineteen, and v2 uses seed nine. Each printed row summarizes two hundred evaluation episodes.

This is representative policy inference, not a new training sweep. Aggregate scientific claims still come from all twenty independently trained seeds. The included command reproduces this execution after installing the locked software environment.

## 13. Fresh v2 execution in 3D

The public simulator also runs the final v2 policies live, directly in the browser. The three-dimensional scene is a visual representation of the grid simulation. Each action is computed from the exported recurrent neural networks; there are no language-model calls.

The three panels compare the same scenario. On the left, the pair receives its learned two-word messages. In the center, both word positions are scrambled. On the right, the same communicating policies hear silence. The geometry, target, and trained weights stay fixed. Messages arrive one step later, and each episode starts with reset recurrent memory.

These are representative demonstrations, not additional benchmark samples. The website separately displays the full twenty-seed results and clearly reports the failed generalization criterion. The code ZIP includes everything needed to run this simulator locally without an account or an API key.

## 14. Useful coordination, with bounded claims

The main finding is that a cue-directed signalling objective supports useful coordination in this split-information task, and the message interventions show that the runner depends on stable message identities.

The extension establishes an important limit: communication success does not automatically produce compositional generalization.

The project connects to the course's agent reinforcement learning and multi-agent systems material. Its contribution is a controlled implementation and evaluation, with the comparisons, negative findings, and limitations reported explicitly.

Claude and ChatGPT or Codex assisted with planning, implementation, testing, and report refinement, as disclosed in the report. I remain responsible for understanding and explaining the work.

Thank you.
