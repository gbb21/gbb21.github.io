# Commenting on the O1 Implementation

## Background
On the release of the [OpenAI O1](https://openai.com/index/learning-to-reason-with-llms/) model, there are a lot of buzz on the internet about how does it create a new scaling law and how does it achieve the *automated chain-of-thought* capability.

Most of the online discussions actually converges into 2 key techniques:
1. RL with [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
2. RL with [Self-Play](https://en.wikipedia.org/wiki/Self-play)

As you can see both techniques are used in [Alpha-Zero](https://en.wikipedia.org/wiki/AlphaZero) project, and they are key to the multi-agent games. Nevertheless, it is not immediately clear on how would they be applied to *OpenAI O1* to achieve the state-of-the-art logic reasoning capabilities. In the rest of this note, I will share some of my thought of over one of the possibilities.

## Fundamentals
Before we start, let's first talk about what are MCTS and Self-Play, and how would they usually being used in RL.

MCTS is commonly used in RL to help solve the credit assignment problem - estimate the current action's accumulated return, aka $Q(S, a)$, by using *planning*.
For instance, for a given state, MCTS will use [multi-armed-bandit] (https://en.wikipedia.org/wiki/Multi-armed_bandit) to virtually and recursively expand the next possible actions - next possible state - next next possible actions ...
and so on, until it reach some final state (or it will force itself stop at some point and apply *roll-out* to estimate the rest of the moves), and back-propagate the rewards collected along the way to the root node.
By using this method, RL agent is able to attribute the reward happens at the very later stage of the multi-step process into the immediate next step.

This method successfully helped AlphaZero planned its best next moves when play Go games.

Self-Play is a technique to let RL-agents to complete against themselves in the *multi-agent-game-play* environment.
The RL-agent under this settings will play 2 roles with adversarial goals (rewards).
By let them explore the completing environment against each other, they will finally reach the [min-max equilibrium](https://en.wikipedia.org/wiki/Minimax) that expose the minimal opportunity to opponents to win.
By doing that, even without human expert guidance, RL-agents will still achieve the best possible performance.

Other than the *zero-sum-two-player-games*, the Self-play is also a commonly observed pattern in the [Generative Adversarial Network](https://en.wikipedia.org/wiki/Generative_adversarial_network) training.
It can be used to **autonomously** improve the *generator* performance by using a *discriminator* to evaluate the performance of the generated result. GAN was initially developed for image generation, but later being employed in LLM training as well to help LLM generate text sequence closer to human preferences.


## How does O1 Benefit from MCTS
The observed O1 model will generate a super long *hidden chain-of-thought* before really answering user's question. The conversation looks like following
>> <|User|> User prompts
>> <|Agent|> [Thought] thought step 1 \n thought step 2 \n ... \n thought step n [/thought] begin the answer here ....

It's clear that, given a good / correct *thought* process, model will achieve a good result on solving the math problems. Previously, OpenAI had a research about using [process supervision](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/) to help the mathematical reasoning.

This approach is all good for solving a small fixed domain problem, except human labeler become the bottleneck for the model to grow (to multiple domains), and we loss the [scaling law](https://arxiv.org/abs/2001.08361) for language model training.
This also prevent the model to achieve [super-human performance](https://openai.com/index/introducing-superalignment/).

RL comes to a save here - when we treat LLM as the RL-policy, RL will let the LLM figure out how to generate its "intermediate thought steps" by itself, as long as human give it some reward at the end (when model generate the correct final answer).
In a [grounded](https://techcommunity.microsoft.com/t5/fasttrack-for-azure/grounding-llms/ba-p/3843857) environment, model is also able to get the final reward by itself, and learn completely without human supervision.

The LLM policy trained under RL setting will learn to "explore" different "thought steps" on its own. During the inference time, it can effectively perform the "search" process in the problem solution space even without needs of the [value function](https://en.wikipedia.org/wiki/Value_function), which makes the LLM policy can be inferenced just like any other LLMs (without inference time overhead).
Of course, it is also possible to add inference time RL (which is a similar concept to [test time training](https://arxiv.org/abs/2407.04620)) as well to further improve the model performance if we have enough time budget during inference.

Above plan for training O1 sounds all perfect, except in reality the RL policy is extremally hard to train, because
1. thought (search) process is very long
1. we have extremely sparse reward (only at the end of the generation)

Other than applying a lot of RL tricks like *reward engineering* to create denser rewards, one important trick to help the policy (LLM) to achieve *best next step" is to perform planning by using MCTS. MCTS helps optimize the exploration process (by using UCB), and speed up the "reward propagation". By aligning the policy (LLM) next step prediction into the MCTS next step estimation, we can create a policy with better planning capabilities.

## How does O1 Benefit from Self-Play
From the above analysis, the O1 model's reasoning capability seems not requiring "self-play", as it can use RL with MCTS to achieve *automated chain-of-thought*. However, this is based on 2 important assumptions -
1. human is able to give final reward to the model output (or ground the model to the world)
2. we do not care if the "thought process" aligned with human preference.

The first assumption is easy to achieve if we are asking questions with closed form answers (e.g. math, science questions), but usually it is hard to define a perfect final reward function for real world open ended questions.

Let me explain a bit more about the second assumption -
Because the "thought process" is completely generated by the model RL process, it is quite possible to create a path that *reasonable* to machine, but not understandable by human.

For example, when we ask model to answer
>> If earth is a sphere ?
>> [thought] 1. we can reach escape velocity to enter Earth orbit; 2. we can observe with our eyes the Earth is spinning underneath [/thought] so earth is a sphere.

Though the answer is good, but the thought process is not aligned with human reasoning (but it may align with the LLM reasoning), as the RL's goal is to create a path to correct answer, but it doesn't really care about how good is the path aligning with human preference. The path could be self-consistent, but not aligned with human's reasoning process.
What's worse, sometimes the unsupervised thought process could even reveal risky information, thus OpenAI decide to hide the reasoning process for now.

Despite in reality, LLM policy usually generate thought process most align with human understanding (as it was pre-trained with human language dataset), it is still beneficial to have a supervision model to help automatically align LLM's thought process.

Yes, you guess it right. Here is how *Self-Play* comes to rescue.

The approach is following the *adversarial training* framework to create a *critic model* that can help distinguish the "human thought process" and the "model thought process". The critic model is can also help to generate the "final reward" for open ended questions.
When we iteratively use the *critic model* to improve the the *generation policy (LLM)* performance, the critic model can also get updated to distinguish the "human thought process" and the "improved model thought process".  It will align the model both final answer and thought process with human preference without needs of human judges.

This sounds like a very promising approach, but may still in the research stage. Based on what have been published by OpenAI O1, I personally do **not** believe it has been applied in the O1 model.

# Disclaimer

The writing above is based on limit observation from O1's behavior and author's personal understanding. It is not supported by any OpenAI's officials, and you should use the information as your own risk.
