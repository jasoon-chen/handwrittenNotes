#  The Prediction Objective

In the tabular methods, a continious measure of prediction quality was not necessary because the learned value function would equal the true value function exactly. But with approximation, an update at once state affects many others meaning that it is not possible to get the values of all states exactly correct. We must specifiy a state distrubtion representing how much we care about the error in each state $s$.

$ \mu(s) \ge 0, \sum_{s}\mu(s)=1$

This is how much we care about the error in each state $s$.

With the error in the state $s$, we need to utilize the square of the difference between the approximate value $\hat{v}(s,w)$ and the true value $v_{\pi}(s)$

The Mean Squared Value Error denoted as $\overline{VE}(w)=\sum_{s\in S}\mu(s)[v_{\pi}(s)-\hat{v}(s,w)]^2$

# Linear Methods

One of the most important function approximation is that which the approximation function $\hat{v}(\dot,w)$ is a linear function of the weight vector w. Linear methods approximate state-value function by the innterproduct between the weights $w$ and the feature vector, $x(s)$:

$\hat{v}(s,w)=w^Tx(s)=\sum_{i=1}^{d}w_{t}x_{i}(s)$

The gradient of this with respect to w is:

$\nabla\hat{v}(s,w)=x(s)$

# Coarse Coding
One kind of representation for this case is made up of features corresponding to circles in state space, as shown to the right. If the state is inside a circle, then the corresponding feature has the value 1 and is said to be present; otherwise the feature is 0 and is said to be absent. This kind of 1–0-valued feature is called a binary feature. Given a state, which binary features are present indicate within which circles the state lies, and thus coarsely code for its location. Representing a state with features that overlap in this way (although they need not be circles or binary) is known as coarse coding.
![alt text](../RL/images/image.png)

# Tile Coding
Tile coding is a form of coarse coding for multi-dimensional continuous spaces that is flexible and computationally efficient. It may be the most practical feature representation for modern sequential digital computers.
![alt text](../RL/images/image-1.png)

# Stacking Features
In SARSA, we need to move from state-values to action-values. Such as in this example:

$q_{\pi}(s,a)\approx\hat{q}(s,a,w) = w^{T}x(s,a)$

One way to do this is to have a state for each action or also known as **stacking the features**.

Let's assume there are 4 features:

$x(s)=[x_{0}(s),x_{1}(s),x_{2}(s),x_{3}(s)]^T$

With these actions:

$A(s) = {a_{0},a_{1},a_{2}}$

To represent this by learning both the states and actions you can rewrite the states as:

$x(s)=[\color{blue}x_{0}(s),x_{1}(s),x_{2}(s),x_{3}(s),\color{red}x_{0}(s),x_{1}(s),x_{2}(s),x_{3}(s),\color{green}x_{0}(s),x_{1}(s),x_{2}(s),x_{3}(s)\color{black}]^T$ Now the feature vector has 12 components.

# Episode Semi-Gradient Control

The general gradient descent update for action-value prediction is:

$w_{t+1} = w_{t} + \alpha[U_{t}-\hat{q}(S_{t},A_{t},w_{t})]\nabla\hat{q}(S_{t},A_{t},w)$

The update for the one-step Sarsa method is:

$w_{t+1} = w_{t} + \alpha[R_{t+1}+\gamma\hat{q}(S_{t+1},A_{t+1},w_{t+1} - \hat{q}(S_{t},A_{t},w_{t})]\nabla\hat{q}(S_{t},A_{t},w)$

To form control methods, we need to couple such action-value prediction methods with techniques for policy improvement and action selection. Suitable techniques applicable to continuous actions, or to actions from large discrete sets, are a topic of ongoing research with as yet no clear resolution. On the other hand, if the action set is discrete and not too large, then we can use the techniques already developed in previous chapters. That is, for each possible action a available in the current state $S_{t}$, we can compute $\hat{q}(S_{t},a,w_{t})$ and then find the greedy action. Policy improvement is then done (in the on-policy case treated in this chapter) by changing the estimation policy to a soft approximation of the greedy policy such as the "-greedy policy. Actions are selected according to this same policy.
![alt text](../RL/images/image-2.png)