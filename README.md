# lmenv

Environment implementations for GRL games (Tetris and Sokoban). This repository provides self-contained, deterministic game environments suitable for reinforcement learning and language model training.

## Overview

`lmenv` contains implementations of classic puzzle games adapted for use in the GRL (Game Reinforcement Learning) framework. The environments are designed to be:

- **Deterministic**: Reproducible game states with seed-based generation
- **Text-based**: Human-readable game representations for language model interaction
- **Self-contained**: Minimal external dependencies
- **RL-ready**: Compatible with standard RL frameworks and Gymnasium interfaces

## Supported Games

### Sokoban
A classic box-pushing puzzle game where the player must push boxes onto target locations.

- **Gameplay**: Navigate a warehouse, push boxes onto marked targets
- **Difficulty**: Configurable puzzle complexity (room size, number of boxes)
- **Reward Structure**: Positive rewards for successful moves, penalties for invalid actions
- **State Representation**: Text-based grid with symbols for walls, boxes, targets, and player

### Tetris
The classic falling-block puzzle game.

- **Gameplay**: Rotate and place falling tetrominoes to clear lines
- **Difficulty**: Configurable board size and piece generation
- **Reward Structure**: Points for line clears, penalties for game over
- **State Representation**: Text-based board representation


## Integration with GRL

These environments are designed to work with the [GRL framework](https://github.com/lmgame-org/GRL). 

## Integration with NeMo Gym

These environments are in the process of being integrated into [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) as resource server environments for RL training. 

## Citation

If you use `lmenv` in your research, please cite:

```bibtex
@software{lmenv2025,
  title={lmenv: Game Environments for GRL},
  author={lmgame-org},
  year={2025},
  url={https://github.com/lmgame-org/lmenv}
}
```

## Related Projects

- [GRL](https://github.com/lmgame-org/GRL) - Game Reinforcement Learning framework
- [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) - Integration of these environments
- [gym-sokoban](https://github.com/mpSchrader/gym-sokoban) - Original Sokoban implementation
