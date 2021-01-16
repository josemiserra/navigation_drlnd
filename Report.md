<br />
  <h3 align="center">Project 1 Navigation using DQN and Unity ML</h3>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


## About The Project



## Variation of parameters on basis DQN 

First, we adapted the LunarLander-v2 agent and model from the  [DQN lesson at Udacity](https://youtu.be/MqTXoCxQ_eY).
The files can be downloaded from the  [RL nanodegree repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn).

The adaptation requires simply to change the OpenAI Gym environment by the Unity ML environment and adapt some of the parameters. The goal now is to achieve a value more than 13 as an average of the last 100 episodes, in less than 1800 episodes.

This is achieved easily by the original DQN around 500 episodes.  Since the threshold to achieve has been already achieved, our goal would be now to achieve the same results but with less time. Since the first 100 episodes the network is still learning and starts from 0, it is unlikely to achieve an average of 13 then.Thus we will consider a success if we achieve the average in 200 episodes.

<table>
  <tr>
    <th>&nbsp;</th>
    <th>Episodes</th>
    <th>Mean +/- std</th>
  </tr>
  <tr>
    <td>Vanilla DQN</td>
    <td>473, 505, 516 </td>
    <td>498 +/- 18</td>
  </tr>
  <tr>
    <td>Age</td>
    <td>16</td>
    <td>9</td>
  </tr>
  <tr>
    <td>Owner</td>
    <td>Mother-in-law</td>
    <td>Me</td>
  </tr>
  <tr>
    <td>Eating Habits</td>
    <td>Eats everyone's leftovers</td>
    <td>Nibbles at food</td>
  </tr>
</table>
## Usage


## Extensions to the original DQN




## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)


## References
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)




