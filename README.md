# voxelgym2D
A gym environment for voxel/grid based reinforcement learning for path planning.

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

## Changelog
### 0.1.0
- Initial release of voxelgym 2D environments tested with stable_baselines 3 (v1.6.2) and python 3.8

## TODO
- [x] Add 2D environments
- [ ] Test with gym 0.26.2
- [ ] Add documentation

## Known issues
- [ ] Currently only supports gym==0.21.0 :sad:,hence setuptools==65.5.0 is required to install gym.

## Contributing
Contributions are welcome! Please open an issue or a pull request.

## References
- [OpenAI Gym](https://arxiv.org/abs/1606.01540)
- [Stable Baselines 3](http://jmlr.org/papers/v22/20-1364.html)
