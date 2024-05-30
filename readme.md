# Playing Atari with Bayesian Deep Expected Sarsa

## Overview

This project explores the use of an on-policy algorithm, Deep Expected SARSA, for playing Atari games. The objective is to address the challenges associated with Deep Q-Learning, such as instability and overestimation, by integrating Bayesian layers into the neural network architecture. This approach aims to enhance decision-making in stochastic environments.

## Key Features

- **Algorithm**: Implements Deep Expected SARSA to improve learning stability and performance by considering the expected value of future rewards under the current policy.
- **Bayesian Framework**: Incorporates Bayesian layers into the neural network using Bayes-Backprop, which enables more robust decision-making and uncertainty quantification.
- **Neural Network Architecture**: Utilizes a convolutional neural network with Bayesian layers, based on the architecture proposed in "Playing Atari with Deep Reinforcement Learning."
- **Exploration Strategy**: Employs an $\epsilon$-greedy strategy with a decaying factor, optimized through various parameter sets for efficient exploration.

## Algorithm

Deep Expected SARSA updates the Q-values using the following rule:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \sum_{a'} \pi(a'|s') Q(s',a') - Q(s,a)]
$$

### Neural Network Design

The neural network processes environment state data through several convolutional layers, followed by fully connected layers. Bayesian layers are integrated to leverage the stochasticity and enhance the exploration strategy.

### Pseudo Code

```python
Initialize Q(s, a) arbitrarily
Initialize empty replay buffer D
For each episode:
    Initialize state S
    Choose action A using policy derived from Q
    For each step in episode:
        Take action A, observe reward R and next state S'
        Add (S, A, R, S') to D
        If size of D is sufficient:
            Compute expected value over next actions
            Update Q-value
            Clear replay buffer D
        S ← S'
```
