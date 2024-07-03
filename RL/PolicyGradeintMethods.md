# Introduction

Most methods have been action-value methods, meaning that they learn the values of actions and then select actions based on there estimated action values. This means that there policies would not even exist without the action-value estimates. There can be parameterized policy that can select actions without consulting a value function. The notation $\theta\in R^d$ for the policy paramter vector. 

# Policy Approximation and its Advantages

If the action space is not to discrete and not too large, then a natural and common kind of paramterization is to form a paramterized numerical preferences $h(s,a,\theta)\in R$ for each state action pair. The actions with the highest preferences in each state are given the highest probabilities of being selected, for example, according to an exponential soft-max distribution:

$pi(a|s,\theta)=\frac{e^{h(s,a,\theta)}}{\sum_{b}e^{h(s,b,\theta)}}$ This kind of policy is called soft-max in action preferences. The action preferences themselves can be parameeriozed arbitarily. For example, they might be computed by a deep artificial neural network where $\theta$ is the vector of all the connection weights of the network. Or the preference could be linear in features such as:
$h(s,a,\theta)=\theta^Tx(s,a)$

Advantages of parameterizing policies according to the soft-max in action preferences is that the approximate policy can approach a deterministic policy whereas episilon greedy action selection over action values there is always an epsilon probability of selecting a random action. Second advantage of parameterizing policies according to soft-max is that it enables the selection of actions with arbitrary probabilities. In problems with significant function approximation, the best approximate policy may be stochastic. **Action-value methods have no natural way of finding stochastic optimal policies, where policy approximating methods can**

The choice of policy parameterization is sometimes a good way of injecting prior knowledge about the desired form of the policy into the reinforcement learning system. This is often the most important reason for using a policy-based learning method.

# The Policy Gradient Theorem

With function approximation, it may seem challenging to change the policy parameter in a way that ensures improvement. The problem is that performance depends on both the action selections and the distribution of states in which those selections are made, and that both of these are a↵ected by the policy parameter. Given a state, the eect of the policy parameter on the actions, and thus on reward, can be computed in a relatively straightforward way from knowledge of the parameterization. But the eect of the policy on the state distribution is a function of the environment and is typically unknown. How can we estimate the performance gradient with respect to the policy parameter when the gradient depends on the unknown e↵ect of policy changes on the state distribution? Fortunately, there is an excellent theoretical answer to this challenge in the form of the policy gradient theorem.

$\nabla J(\theta) \propto \sum_{s}\mu(s)\sum_{a}q_{\pi}(s,a)\nabla\pi(a|s,\theta)$