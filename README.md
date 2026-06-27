
# LLM-Enhanced Genetic Algorithm Optimization Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains **two implementations** of a Genetic Algorithm (GA) enhanced with **DeepSeek Large Language Model (LLM) guidance**, designed for solving complex continuous optimization problems. The LLM assists in generating superior offspring and mutation vectors, leading to faster convergence and better solutions.

---

## Implementations

| Implementation | Objective | Dimensions | Key Focus |
|:---|:---|:---:|:---|
| **LLM-GA-Rastrigin** | Minimize Rastrigin function (converted to maximization) | Configurable (default: 10) | Classic benchmark optimization with LLM-guided crossover/mutation |
| **LLM-GA-FAS** | Maximize `\|h(yt, zt, yr, zr)\|` (Fluid Antenna System) |  Fixed: 4 (`yt`, `zt`, `yr`, `zr`) | Engineering problem with historical population data in LLM prompts |
---

## Key Features

- **Hybrid Intelligence** – Combines traditional GA operators with LLM-generated candidates.

- **Adaptive Operators** – Dynamically adjusts mutation and crossover rates based on convergence progress.
- **Robust API Handling** – Built-in retries, response validation, and graceful fallback to traditional methods when the API fails.
- **Diversity Maintenance** – Detects premature convergence and injects fresh random individuals.
- **Comprehensive Logging** – Tracks best/average fitness, population diversity, and API failure counts per generation.

---

## Installation & Dependencies

**Python 3.8+** is required. Install the necessary libraries:

```bash
pip install numpy openai
```


## DeepSeek API Key Configuration

To enable LLM guidance, you need a valid DeepSeek API key. Provide it via:

-   **Environment variable** (recommended):
    

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```
-   **Direct assignment** in the code (not recommended for public repositories):
    

```python
ga = EnhancedDeepSeekGA(..., deepseek_api_key="your-api-key-here")
```
> **Warning:** Never commit your API key to version control. The code reads from the environment variable `DEEPSEEK_API_KEY` by default.

## Usage

Each script is self-contained. Simply run:
```bash
# Optimize Rastrigin function
python LLM-GA-Rastrigin.py
# Optimize FAS function
python LLM-GA-FAS.py
```
### Enabling / Disabling LLM Guidance

To compare performance, you can turn off LLM assistance:
```python
ga = EnhancedDeepSeekGA(..., use_deepseek=False)
```
## Configuration Parameters

Common parameters available in both implementations:

| Parameter | Description | Rastrigin Default | FAS Default |
|:---|:---|:---:|:---:|
| `objective_func` | Objective function (must be maximization form) | Required | Required |
| `dim` | Dimensionality of the problem | 10 | 4 |
| `population_size` | Number of individuals | 50 | 30 |
| `generations` | Number of evolutionary generations | 100 (in `evolve()`) | 20 |
| `mutation_rate` | Base mutation probability | 0.1 | 0.15 |
| `crossover_rate` | Base crossover probability | 0.9 | 0.85 |
| `use_deepseek` | Enable LLM guidance | `True` (if API key exists) | `True` (if API key exists) |
| `deepseek_temperature` | LLM output randomness (0–1) | 0.4 | 0.3 |

## Output & Interpretation
During evolution, you will see per-generation logs like:
```text
Gen 012: Best=1.2345 Avg=0.9876 Mut=0.12 Div=0.345 API Fail=0/10
```
| Field | Meaning |
|:---|:---|
| **Best** | Current best fitness (in maximization sense) |
| **Avg** | Average fitness of the population |
| **Mut** | Current mutation rate (adaptively adjusted) |
| **Div** | Population diversity (mean standard deviation across dimensions) |
| **API Fail** | Number of failed LLM API calls |

At the end, the script prints:

```text
Best solution: [0.0001, -0.0002, ...]
Best fitness: 0.001234 (original objective value)
```
## How LLM Guidance Works

The algorithm constructs **detailed prompts** that include:

-   The objective function expression and variable bounds.
    
-   Current parent individuals (for crossover) or the individual to mutate.
    
-   Population statistics (best fitness, diversity, generation number).
    
-   Historical summary of previous generations (in the FAS version).
    

The LLM then returns new candidate vectors, which are clipped to the valid range, rounded to 4 decimal places, and integrated into the population. This approach helps the algorithm **escape local optima** and **converge faster** compared to purely random operators.

## Customization & Extension

-   **New objective functions** – Write your own `objective_func(x)` that returns a float (maximization form). For minimization, simply negate the value.
    
-   **Different variable ranges** – Modify `range_min` and `range_max` in the class, and update the prompt accordingly.
    
-   **Higher dimensions** – Change `dim` – the prompt generator adapts automatically.
    
-   **Tuning API behavior** – Adjust `deepseek_temperature`, `max_retries`, and `max_api_failures` to balance creativity and stability.

## Important Notes

-   API calls introduce latency and rely on network stability; the framework includes retry and exponential backoff.
    
-   LLM-generated vectors are always clipped to the valid range and rounded to 4 decimal places.
    
-   If API failures exceed `max_api_failures`, the system automatically disables LLM guidance and continues with traditional operators.
    
-   This is a research prototype – for production use, add more extensive error handling and validation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for:

-   Bug fixes and performance improvements.
-   Additional optimization test functions.
    
-   Improved prompting strategies for the LLM.
    
-   Better adaptive control mechanisms.
