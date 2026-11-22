# Product Direction

Caliper is a live instrument bench for machine learning models. An ML engineer and an AI agent build interactive instruments on a running model, during training or inference, to see what the model does, understand why, and change it. This page states the direction that the roadmap follows. The roadmap graph and its pillars show the features on the way.

## Users

Caliper is for ML engineers and researchers who work in PyTorch on their own Mac or NVIDIA machine. They debug training, interpret what a model has learned, explore data and embeddings, and show a model to other people.

## Workflow

The engineer connects a model with one line in the training script, `caliper.watch(model)`, or opens the model directly in Caliper. Caliper then reads the model's tensors on the GPU while it runs, without a copy and without slowing the training.

The engineer asks a question in plain words, in Caliper or in a coding agent, such as "why does the loss spike?" or "what does this attention head look at?". The agent builds an instrument for that question in seconds. An instrument holds plots, 3-D views, visuals that the GPU computes from the model's tensors, and controls. The engineer can also build or change an instrument by hand.

The engineer then works with the instrument. A click, a selection, a step through training, a new input, a pause or a changed parameter goes back to the model, and the model responds while it runs. Thus the engineer tests an idea instead of only looking at a chart. Finally, the engineer saves the instrument and uses it again on the next run or the next model, or shares it.

## Distinction

An instrument is live, so it shows the model while it runs, not a log after the run. It reads tensors on the GPU, so training keeps its speed. It works in two directions, so the engineer steers the model as well as watches it. An agent builds it on demand, so nobody writes a dashboard by hand. It runs in the Caliper app and in a browser.

Caliper is not an experiment tracker, a plotting library, a game engine or a training framework.

## Example Sessions

| Goal | Session |
|---|---|
| Debug training | An attention-entropy view over the training steps shows heads that collapse before a loss spike. The engineer lowers the learning rate and watches the run recover. |
| Interpret a model | One attention head shows its pattern on a sentence and the tokens it promotes. A control turns the head off, and the output changes. |
| Explore data and outputs | An embedding map of the dataset shows each class in its own colour, and a click on a point opens its example. A world model's predicted frames show beside the real frames, with a control for the next action. |

## Priorities

The roadmap puts four capabilities first. An agent builds an instrument live, with views, controls and events that come back to it. The GPU computes visuals from the model's tensors, including programs that the agent writes. The engineer steers and intervenes: pause, parameters, switching off heads and patching activations. Instruments are saved and used again. Recordings, other tensor frameworks, Windows parity and packaging come after these four.
