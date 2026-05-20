# Santa 2025 — Christmas Tree Packing Challenge

> **Kaggle Competition — Santa 2025**

A combinatorial optimisation challenge to pack as many Christmas trees (irregular polygons) as possible into a fixed bounding box without overlap. This repository contains the full solution pipeline including Python exploration notebooks and a high-performance C++ optimiser.

---

## Competition Overview

| Detail | Value |
|---|---|
| **Host** | Kaggle / Google LLC |
| **Series** | Santa 2025 |
| **Task** | Combinatorial optimisation (2D packing) |
| **Metric** | Number of trees packed (maximise) |
| **License** | CC BY 4.0 |

**Competition page:** [Kaggle Santa 2025](https://www.kaggle.com/competitions/santa-2025)

---

## Project Structure

```
.
├── Backups/              # Saved solution backups during optimisation runs
├── Data/                 # Competition data (train/test CSVs)
├── Kraggle/              # Kaggle notebook exports and experiments
├── santa_optimizer.cpp   # Main C++ optimiser (Simulated Annealing + Local Search)
├── submission.csv        # Best submission file
├── test.csv              # Test dataset
└── readme.md             # This file
```

---

## Approach

### Python Exploration
Initial exploration and baseline strategies were developed in Jupyter notebooks (see `Kraggle/` folder), covering:
- Tree polygon parsing and visualisation
- Greedy placement baselines
- Ensemble strategies across multiple solver runs

### C++ Optimiser (`santa_optimizer.cpp`)
The main solver is a high-performance C++ program combining multiple optimisation techniques:

- **Simulated Annealing** — with translation, rotation, and swap move types for broad search
- **Local Search** — squeeze (shrink bounding box), compaction (push trees closer), and fine-tuning passes
- **OpenMP Parallelisation** — explores multiple solution paths simultaneously across CPU cores
- **Continuous Improvement Loop** — reloads the best known solution, attempts improvements, and saves timestamped backups

---

## Building & Running the C++ Optimiser

Requires a C++17 compiler. OpenMP is recommended for parallelisation.

**Option A: GCC (with OpenMP)**
```bash
# Install GCC if needed (macOS)
brew install gcc

# Compile (replace g++-15 with your GCC version)
g++-15 -O3 -march=native -std=c++17 -fopenmp -o santa_optimizer santa_optimizer.cpp

# Run
./santa_optimizer
```

**Option B: Clang (single-threaded)**
```bash
g++ -O3 -march=native -std=c++17 -o santa_optimizer santa_optimizer.cpp
./santa_optimizer
```

> **Note:** Place `submission.csv` (current best solution) in the same directory before running. The optimiser will read it as a warm start and save improved solutions to `Backups/`.

---

## Tech Stack

![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-Parallel-green?style=flat)
