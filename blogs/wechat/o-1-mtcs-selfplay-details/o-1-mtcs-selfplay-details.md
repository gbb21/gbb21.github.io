# Commenting on the O1 Implementation

## Background
On the release of the [OpenAI O1](https://openai.com/index/learning-to-reason-with-llms/) model, there are a lot of buzz on the internet about how does it create a new scaling law and how does it achieve the *automated chain-of-thought*.

Most of the online discussions actually converges into 2 key techniques:
1. RL with [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
2. RL with [Self-Play](https://en.wikipedia.org/wiki/Self-play)

As you can see both techniques are used in [Alpha-Zero](https://en.wikipedia.org/wiki/AlphaZero) project for playing multi-agents game.
But it is not immediately clear on how would they be applied to *OpenAI O1* to achieve the state-of-the-art logic reasoning.
In the rest of this note, I will share some of my thoughts over one of the possibilities.

## Fundamentals
Before we start, let's first talk about what are MCTS and Self-Play, and how would they usually be used in RL.

MCTS is commonly used in RL to help solve the [credit assignment problem](https://ai.stackexchange.com/questions/12908/what-is-the-credit-assignment-problem) - estimate the current action's accumulated return, aka $Q(S, a)$, by using multi-steps planning.
For instance, for a given RL state, MCTS will use [multi-armed-bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) to virtually and recursively expand the next possible actions, then next possible state, then next next possible actions ...
and so on, until it reaches some final state (or it will stop at some depth and apply *roll-out*), and back-propagate the rewards collected along the way back to the root node.
By using this method, RL agent is able to attribute the reward happens at the very later stage of the multi-step process into the immediate next step of the current state.

This method successfully helped AlphaZero plan its best next moves when playing Go games.

On the other hand, Self-Play is a technique have RL-agents compete against themselves in a *game-play* environment.
The RL-agents under this setting will play two roles with adversarial goals.
By fighting against each other, they will finally reach the [min-max equilibrium](https://en.wikipedia.org/wiki/Minimax) which minimize the chance for the opponent to win.
By performing Self-Play, RL-agents can still achieve the best possible performance even without human expert's guidance.

Besides *zero-sum-two-player-games*, the Self-Play is also a commonly observed pattern in the [Generative Adversarial Network (GAN)](https://en.wikipedia.org/wiki/Generative_adversarial_network) training.
It is used to **autonomously** improve the *generator* performance by employing a *discriminator* to evaluate the performance of the generated result. GAN was initially developed for image generation, but later being used in LLM training as well to help LLM generate text sequence align with human preferences.

## How does O1 Benefit from MCTS
The observed O1 model will generate a super long *hidden chain-of-thought* before really answering user's question. The conversation looks like following
> <|User|> User prompt
>
> <|Agent|> [Thought] thought step 1 \n thought step 2 \n ... \n thought step n [/thought] begin the answer here ....

It has been observed that, with a good *thought* process, LLM can achieve a good result on solving math problems.
Previously, OpenAI had a research about using [process supervision](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/) to help the mathematical reasoning.

*Process supervision* is good for solving a small fixed domain problem, but human labelers quickly become the bottleneck for models to grow, and we also lose the [scaling law](https://arxiv.org/abs/2001.08361) for language model training.
It also prevent the model achieving [super-human performance](https://openai.com/index/introducing-superalignment/).

RL comes to a save here - when we treat LLM as the RL-policy, RL will let the LLM itself figure out how to generate its thought processes, as long as human give final rewards at the end (when model generate the correct final answer).
In a [grounded](https://techcommunity.microsoft.com/t5/fasttrack-for-azure/grounding-llms/ba-p/3843857) environment, model can get the final reward by itself, and learn completely without human supervision.

The LLM policy trained under RL setting will learn to "explore" different "thought steps" on its own. During the inference time, a well trained policy can "try out" different reasoning steps without needing a [value function](https://en.wikipedia.org/wiki/Value_function), which allows the LLM policy be inferenced just like any other LLMs (without inference time overhead).
Of course, it is also possible to add inference time RL (which is a similar concept to [test time training](https://arxiv.org/abs/2407.04620)), to further improve the model performance when we have enough time budget during inference.

Above plan for training O1 sounds all perfect, except in reality the RL policy is extremally hard to train, because
1. the thought process is very long
1. we have extremely sparse reward (only get reward at the very end of the generation)

Other than applying a lot of RL tricks like *reward engineering* to create denser reward function, one important trick that helps the policy (LLM) achieve best *next step prediction" is to planning by using MCTS. MCTS helps optimize the exploration process (by using UCB), and speed up the "reward back propagation".

In the training, we also align the policy (LLM) next step prediction with the MCTS next step estimation, which creates a policy with better planning capabilities.

## How does O1 Benefit from Self-Play
From the above analysis, the O1 model's reasoning capability seems not requiring "self-play", as it can use RL with MCTS to achieve *automated chain-of-thought*. However, this is based on 2 important assumptions -
1. human is able to give final reward to the model final output (or ground the model to the world)
2. human does not care if the model "thought process" aligned with human preference.

The first assumption is easy to achieve if we are asking questions with closed form answers (e.g. math, science questions, as the current O1 work best), but it is much harder to define a perfect final reward function for real world open ended questions.

Let me also explain a bit more about the second assumption -
Because the "thought process" is completely generated by the model RL process, it is possible to create a path that *reasonable* to machine, but not understandable by human.

For example, when we ask model to answer
> If earth is a sphere ?
>
> [thought] 1. we can reach escape velocity to enter Earth orbit; 2. we can observe with our eyes the Earth is spinning underneath [/thought] so earth is a sphere.

Though the final answer is correct, but the thought process is not aligned with human reasoning, as the RL's goal is to create a path to correct answer, but it doesn't care about how good is the path aligning with human preference.
The path could be self-consistent, but not aligned with human's reasoning process.
What's worse, sometimes the unsupervised thought process could even reveal dangerous information, thus OpenAI decide to hide the reasoning process for now.

Although in reality, LLM policies usually generate thought processes mostly aligned with human (as they were pre-trained with human language dataset), it is still beneficial to have a supervision model that enforce the alignment.

Yes, you guess it right. *Self-Play* can enforce the alignment.

We follow the GAN framework to create a *critic model* that can help distinguish the "human thought process" and the "model thought process". The critic model can also help to generate the "final reward" for open ended questions.

When we iteratively use the *critic model* to improve the the LLM performance (to generate a improved thought process), the critic model is also get updated to distinguish the "human thought process" and the "improved model thought process". It will force the LLM generate thought process and final answer very close to human preference.

Despite sounds like a very promising approach, it is still in the research stage.
Based on what have been published by OpenAI O1, I personally do **not** believe it has been applied in the O1 model.

## Disclaimer

The writing above is based on limit observation from the released O1 behavior and author's personal understanding. It is not supported by any OpenAI's officials, and you should use the information as your own risk.
