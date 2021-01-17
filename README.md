
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

<p align="center">
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Usage</a></li>
    <li><a href="#gettingstarted">Getting Started</a></li>
    <li><a href="#Instructions">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

The environment is a simplified version of the Banana Collector Environment from the Unity ML-Agents toolkit provided by Udacity. 

A single agent that can move in a planar arena, with observations given by a set of distance-based sensors and some intrinsic measurements, and actions consisting of 4 discrete commands (forward, backward, left, right).

A set of NPCs (Non Player Characters) consisting of bananas of two categories: yellow bananas, which give the agent a reward of +1, and purple bananas, which give the agent a reward of -1. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

![Trained Agent][image1]

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. The observations the agent gets from the environment are the agent's linear velocity in the plane (2 values), and a set of 7 ray-based perceptions. 

A ray perception consist of rays shot in certain fixed directions from the agent (sort of a compass like). Each of these perceptions returns a vector of 5 entries each, with the first 4 values consist of a one-hot encoding of the type of object the ray hit (yellow banana, purple banana, wall or nothing). The last value encodes the distance where the object was found in a percentage (0% if no object were found, 100% if the object is in your position). If no object is found then the 4th value is set to 1, and the percent value is set to 0.0.

  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, with a maximum of 300 steps per episode, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


### Getting Started

1. Clone the repo
   ```sh
   git clone https://github.com/josemiserra/navigation_drlnd
   ```
2. If you don't have Anaconda or Miniconda installed, go to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install Miniconda in your computer (miniconda is a lightweight version of the Anaconda python environment). 

3. It is recommended that you install your own environment with Conda. Follow the instructions here: [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). After that, open an anaconda command prompt or a prompt, and activate your environment.
  ```sh
  activate your-environment
  ```
4. Install the packages present in requirements.txt
   ```sh
   pip install requirements.txt
   pip install mlagents
   ```
5. move into the folder of the project, and run jupyter.
   ```sh
  jupyter notebook
   ```

If you want the original material, download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

### Instructions

Follow the instructions in `Navigation.ipynb` for training the agent. After the training, in the last part it will run in test mode.

## More info
Read the report Report.md

## License

Distributed under the MIT License from Udacity Nanodegree. See `LICENSE` for more information.


## Contact

Jose Miguel Serra Lleti - serrajosemi@gmail.com

Project Link: [https://github.com/josemiserra/navigation_drlnd](https://github.com/josemiserra/navigation_drlnd)


