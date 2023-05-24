# averaging neural networks

i decided to extend the [lth framework](https://github.com/facebookresearch/open_lth) from jonathan frankle for my experiments on averaging neural networks initialiazed from a common set of parameters. frankle's framework uses easy-to-understand and clean software practices such as a hyparameter object to describe the parameters of some entity (e.g TrainingHparams), automatic experiment naming based on experiments, callbacks in the training procedure, and my personal favorite, a Step object that encapsulates the idea of epochs/steps. i re-implement most of these functions in this codebase.

my extension is under the workflow name `spawning`. It trains a **parent** neural network while saving model and optimizer checkpoints at every log step. then, it uses these checkpoints to initialize **children** networks and trains each with the same hyperparameters as the parent. once the training runs are complete, it computes averages among the checkpoints, which for my experiments are 1. averages *across* children at the same time step (called avg_across) and 2. averages *back* betwen a child and parent (called avg_back).

the directory for a `spawning` workflow is as follows:

```   
├── spawn_hyperparameter_hash
│   ├── parent
│   │   ├── logger
│   │   ├── model*ep*it.pth  
│   │   ├── optim*ep*it.pth  
│   ├── children
│   │   ├── *ep*it
│   │   │   ├── seed*
│   │   │   │   ├── logger
│   │   │   │   ├── model*ep*it.pth 
│   │   │   │   ├── optim*ep*it.pth 
│   ├── avg_back
│   │   ├── *ep*it
│   │   │   ├── seed*
│   │   │   │   ├── logger
│   │   │   │   ├── model*ep*it.pth 
│   │   │   │   ├── optim*ep*it.pth 
│   ├── avg_across
│   │   ├── *ep*it
│   │   │   ├── seeds*,*,...
│   │   │   │   ├── logger
│   │   │   │   ├── model*ep*it.pth 
│   │   │   │   ├── optim*ep*it.pth 
```

there are also 2 other directories that compute other metrics of a spawning runner instance. the `plane` directory uses descriptions of 3 network checkpoints (such as a parent and 2 of its children at some step) to compute the loss and accuracy over the plane intersecting these 3 checkpoints. similarly, the `distance` directory computes the l_2 distances from a chld to the origin, to its parent, and to another child trained for the same number of steps (sibling).