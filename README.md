# Ultimate Tic-Tac_Toe

An AI agent for **Ultimate Tic-Tac-Toe** (a 3×3 grid of Tic-Tac-Toe boards where each move “sends” the opponent to a specific local board).

This repository contains:
- A playable/searchable **game state implementation**
- A `StudentAgent` that plays using **Minimax + Alpha–Beta pruning**
- The original **mini-project notebook** describing the rules, state representation, and testing harness
- Supporting dataset + figures

## Repository Contents

- `Agent.py`  
  Implements `StudentAgent`, which selects actions via **minimax with alpha–beta pruning** (depth = 3).  
  Uses a heuristic evaluation function that considers:
  - Meta-board win potential (two-in-a-row patterns)
  - Local board tactics (two-in-a-row, center/corner control)
  - Fork creation
  - Mobility / number of still-open local boards
  - “Free move” situations (when the forced target board is already finished)
  - Large terminal win/loss bonus

- `utils.py`  
  Core environment utilities:
  - `State` / `ImmutableState` representation (board is a **3×3×3×3** NumPy array)
  - Legal move checking (`is_valid_action`)
  - Valid move generation (`get_all_valid_actions`)
  - State transitions (`change_state`)
  - Terminal detection + utilities (`is_terminal`, `terminal_utility`)
  - `invert()` helper for swapping player perspective
  - `load_data()` for loading a dataset from `data.pkl`

- `mini-project.ipynb`  
  The assignment notebook/specification with:
  - Game rules and diagrams
  - Technical description of state/action format
  - Example testing loop for running full games against baseline agents

- `data.zip`  
  Zipped dataset (used for training/experiments; the notebook utilities expect `data.pkl`).

- `figures/`  
  Diagrams used by the notebook (board basics, move mechanics, win/tie examples).

## State / Action Format

- **State**: 4D array of shape `(3, 3, 3, 3)`  
  - `(i, j)` selects the local board on the meta-board  
  - `(k, l)` selects the cell within that local board  
  - Values: `0` empty, `1` player 1, `2` player 2

- **Action**: `(i, j, k, l)` (meta row, meta col, local row, local col)

## How to Use

If you want to run experiments locally:
1. Ensure dependencies like `numpy` are installed.
2. Use `mini-project.ipynb` to test the agent and run sample matches.
3. Keep `utils.py` in the same directory so imports work.
4. Unzip data.zip in the same directory as mini-project.ipynb

---

This project focuses on building a strong evaluation function and using minimax search (with alpha–beta pruning) to play Ultimate Tic-Tac-Toe effectively.
