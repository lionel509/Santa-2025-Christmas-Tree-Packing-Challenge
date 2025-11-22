# Santa 2025 - ML Packing System

🎄 **Fully Autonomous Machine Learning System for Christmas Tree Packing Optimization**

## Overview

This is a complete, autonomous ML-powered optimization system for the Kaggle "Santa 2025 – Christmas Tree Packing Challenge". The system automatically generates optimal packing solutions for all 200 puzzles (1-200 trees) and provides real-time visualization through a web interface.

## Features

- **Fully Autonomous**: Zero manual intervention required
- **Hybrid Optimization**: Combines PyTorch RL (PPO) with heuristic algorithms
- **Real-time Visualization**: WebSocket-powered live updates
- **Persistent State**: Auto-saves progress every 5 minutes
- **Collision Detection**: Fast SAT-based polygon collision checking
- **Submission Generator**: Automatic CSV export in correct format

## System Architecture

```
ml_packing_system/
├── app/
│   ├── api/           # FastAPI backend + WebSocket
│   ├── ml/            # PyTorch RL agent (PPO)
│   ├── optimizers/    # Heuristic algorithms
│   ├── geometry/      # Collision detection & bounds
│   ├── state/         # State management & persistence
│   └── models/        # Data models
├── frontend/          # HTML/JS spectator interface
└── data/             # Auto-saved states & submissions
```

## Installation

1. **Install dependencies**:
```bash
cd ml_packing_system
pip install -r requirements.txt
```

2. **Run the system**:
```bash
python run.py
```

3. **Open browser**:
Navigate to `http://127.0.0.1:8000/app`

## Usage

### Starting the System

```bash
# Basic usage (CPU)
python run.py

# Use GPU (if available)
python run.py --device cuda

# Disable ML (heuristics only)
python run.py --no-ml

# Custom host/port
python run.py --host 0.0.0.0 --port 8080
```

### Web Interface

The spectator interface provides:

- **Live visualization** of tree packing
- **Real-time statistics** (score, iterations, improvements)
- **Puzzle selector** to view specific configurations
- **Play/Pause** control for updates
- **Export button** to download submission CSV
- **Activity log** showing optimization progress

### API Endpoints

- `GET /api/state` - System status summary
- `GET /api/puzzle/{n}` - Get specific puzzle (1-200)
- `GET /api/puzzles/all` - Get all puzzle summaries
- `POST /api/save` - Save current state
- `GET /api/export` - Export submission CSV
- `WS /ws/updates` - WebSocket for real-time updates

## Optimization Strategy

### Hybrid Approach

1. **Initialization**: Greedy placement algorithm
2. **ML Optimization**: PPO reinforcement learning agent
3. **Heuristic Refinement**:
   - Local search
   - Simulated annealing
   - Bounding box squeeze
   - Position jittering
   - Rotation sweeps

### Continuous Optimization

The system runs perpetually, cycling through all 200 puzzles:
- Each puzzle gets 50 iterations per cycle
- ML and heuristics are adaptively selected
- Best solutions are automatically saved
- Real-time updates broadcast via WebSocket

## Geometry Engine

### Tree Polygon

15-point polygon representing Christmas tree:
- Trunk: 0.15 × 0.2
- Base tier: 0.7 width
- Mid tier: 0.4 width
- Top tier: 0.25 width
- Tip: 0.8 height

### Collision Detection

- **SAT (Separating Axis Theorem)** for polygon overlap
- **Bounding box pre-filter** for performance
- **Fast matrix operations** using NumPy

### Scoring

Score = s² / n, where:
- s = side length of square bounding box
- n = number of trees

## State Management

### Auto-save

- Saves every 5 optimization cycles
- Creates backup before overwriting
- JSON format for human readability

### Persistence

```
data/
├── puzzle_states.json        # Current state
├── puzzle_states_backup.json # Backup
└── submission.csv            # Export output
```

## Submission Format

The system exports a 20,100-row CSV:

```csv
id,x,y,deg
001_0,s0.123456,s-0.234567,s45.678901
...
200_199,s1.234567,s-2.345678,s90.123456
```

- Exactly 20,100 rows (sum of 1+2+...+200)
- IDs formatted as `{n:03d}_{index}`
- Coordinates prefixed with `"s"`
- 6 decimal places precision
- All positions validated (|x|, |y| ≤ 100)
- Collision-free guarantee

## Machine Learning

### PPO Agent

- **State**: Flattened tree positions + global features (604 dims)
- **Action**: Small dx/dy/ddeg adjustments per tree
- **Reward**: Negative bounding box size + collision penalty
- **Architecture**: 256-dimensional hidden layers

### Training

The agent learns online:
- No pre-training required
- Improves through continuous optimization
- Balances exploration vs exploitation

## Performance

### Optimization Speed

- ~10-20 iterations/second per puzzle
- Full 200-puzzle cycle: ~5-10 minutes
- Continuous improvement over time

### Resource Usage

- **CPU**: 1-2 cores (multi-threaded)
- **RAM**: ~500MB
- **Disk**: ~10MB for saved states

## Troubleshooting

### Import Errors

If you see import errors, ensure you're in the project directory:
```bash
cd ml_packing_system
python run.py
```

### Port Already in Use

Change the port:
```bash
python run.py --port 8080
```

### WebSocket Connection Failed

Check firewall settings or use:
```bash
python run.py --host 0.0.0.0
```

## Development

### Project Structure

All components are modular and can be tested independently:

```python
# Test geometry
from app.geometry import ChristmasTree, check_collision
tree1 = ChristmasTree(0, 0, 0)
tree2 = ChristmasTree(1, 0, 0)
print(check_collision(tree1, tree2))

# Test optimizer
from app.optimizers import initialize_puzzle, local_search
puzzle = initialize_puzzle(10, method='greedy')
optimized = local_search(puzzle, iterations=100)
```

### Extending

Add new heuristics in `app/optimizers/heuristics.py`:

```python
def my_optimizer(state: PuzzleState, **kwargs) -> PuzzleState:
    # Your optimization logic
    return optimized_state
```

Register in `app/optimizers/hybrid.py`.

## License

MIT License - See competition rules for submission requirements.

## Credits

Built for Kaggle Santa 2025 Challenge using:
- FastAPI for backend
- PyTorch for ML
- NumPy for geometry
- WebSockets for real-time updates

---

**Status**: ✅ Ready for autonomous operation
**User Interaction**: 🚫 None required (spectator mode only)
**Submission**: ✅ Automatic generation via `/api/export`
