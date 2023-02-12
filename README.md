# voxelgym2D
A gym environment for voxel/grid based reinforcement learning for path planning.

<div id="solution-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="https://github.com/harisankar95/voxelgym2D/raw/main/resources/solution_1.gif" width="375"/>
      	    </td>
            <td style="padding:10px">
            	<img src="https://github.com/harisankar95/voxelgym2D/raw/main/resources/solution_2.gif" width="375"/>
            </td>
        </tr>
    </table>
</div>

## Results with [SB3](https://github.com/DLR-RM/stable-baselines3) (v1.6.2) : PPO :smile:
Here are the results of training a PPO agent on the `onestep-v0` using the example [here](examples/train_ppo.py). Below you will find the episode reward and episode length over steps during training. As the agent learns, the episode reward increases and the episode length reduces are the agent learns to identify the goal and reach it in the shortest possible path.

<div id="solution-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="https://github.com/harisankar95/voxelgym2D/raw/main/resources/ep_reward.png" width="375"/>
      	    </td>
            <td style="padding:10px">
            	<img src="https://github.com/harisankar95/voxelgym2D/raw/main/resources/ep_length.png" width="375"/>
            </td>
        </tr>
    </table>
</div>


## Installation
```bash
git clone https://github.com/harisankar95/voxelgym2D.git ~/path/to/repo
cd ~/path/to/repo
pip install .

# to aditionally install stable_baselines 3 and pytorch (optional)
pip install .[sb3]
```

or directly from github :smile:

```bash
pip install git+https://github.com/harisankar95/voxelgym2D.git
```

## Development
To install the package in development mode, run the following command in the root directory of the repository:
```bash
pip install -e .[dev]

# to aditionally install stable_baselines 3 and pytorch (optional)
pip install -e .[dev,sb3]
```

## Usage
```python
import voxelgym2D
import gym

env = gym.make("voxelgym2D:onestep-v0")
env.reset()
env.render()
```

## Examples
The examples can be found [here](examples).

## License
This project is licensed under the terms of the [MIT license](LICENSE).

## Documentation
The documentation can be found [here](https://harisankar95.github.io/voxelgym2D/).

## Changelog
### 0.1.0
- Initial release of voxelgym 2D environments tested with stable_baselines 3 (v1.6.2) and python 3.8

## TODO
- [x] Add 2D environments
- [ ] Test with gym 0.26.2
- [ ] Add documentation

## Known issues
- [ ] Currently only supports gym==0.21.0 :neutral_face:,hence setuptools==65.5.0 is required to install gym.

## Contributing
Contributions are welcome! Please open an issue or a pull request.

## References
- [OpenAI Gym](https://arxiv.org/abs/1606.01540)
- [Stable Baselines 3](http://jmlr.org/papers/v22/20-1364.html)
