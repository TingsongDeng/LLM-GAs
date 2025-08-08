LLM-Enhanced Genetic Algorithm Optimization Framework
This repository contains two implementations of a Genetic Algorithm (GA) enhanced with Large Language Model (LLM) guidance from DeepSeek, designed for solving complex optimization problems.

Implementations
1. LLM-GA-Rastrigin
Optimizes the Rastrigin function, a classic benchmark for optimization algorithms:

Objective: Minimize the Rastrigin function (converted to maximization problem)

Dimensions: Configurable (default 10)

Features:

LLM-guided crossover and mutation operations

Adaptive parameter tuning

Premature convergence detection

API failure handling

2. LLM-GA-FAS
Optimizes a function from Fluid Antenna System (FAS) applications:

Objective: Maximize the function |h(u,v)| in FAS

Dimensions: Fixed at 4 (yt, zt, yr, zr)

Features:

Enhanced with historical population data in LLM prompts

Detailed generation logging

Early convergence handling

Key Features
Hybrid Intelligence: Combines traditional GA with LLM guidance

Adaptive Operators: Dynamically adjusts mutation and crossover rates

Robust API Handling: Comprehensive error handling for LLM API calls

Diversity Maintenance: Mechanisms to prevent premature convergence

Detailed Logging: Tracks population diversity and fitness history

Usage
For each implementation:

Install dependencies:

pip install numpy openai

Set your DeepSeek API key:

ga = EnhancedDeepSeekGA(
    ...,
    deepseek_api_key="your_api_key_here"
)

Configuration Options
Common parameters for both implementations:

population_size: Number of individuals in population

mutation_rate: Base probability of mutation

crossover_rate: Probability of crossover

use_deepseek: Enable/disable LLM guidance

deepseek_temperature: Controls LLM creativity (0-1)

Results Interpretation
Output includes:

Generation-by-generation progress

Best and average fitness values

Population diversity metrics

API failure counts

Final best solution and fitness

Requirements
Python 3.8+

NumPy

OpenAI package (for DeepSeek API)

Contributing
Contributions are welcome! Please open an issue or pull request for:

Bug fixes

Additional optimization problems

Improved LLM prompting strategies

Performance enhancements
