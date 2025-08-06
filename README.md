# Self-augmenting Technical Indicator RL

This repository hosts the code and data for our study **“Self-augmenting Technical Indicator with Reinforcement Learning.”**  
The project explores how an agent can unlock the path-dependent information hidden in widely used indicators such as MACD, optimising a portfolio-level metric (e.g., the Sharpe Ratio) rather than step-wise rewards.

## Repository Structure

- **src/**  
  Source code with modules for modelling
- **pyproject.toml**  
  Project dependencies.
- **.gitignore**  
  Standard Python and OS ignore patterns.

## Minimal Usage Example

1. Install dependencies:
  ```bash
  bash install.sh
  ```
2. Process the raw data with:
  ```bash
  data/process_data.ipynb
  ```
3. To run comparison with Online RRL:
  ```bash
  python experiment.py
  ```
4. To run results analysis:
  ```bash
  notebooks/plot_results.ipynb
  ```

## Reference

For the complete description of the method, mathematical details, and experiments, see our article THE REFERENCE TO ADD