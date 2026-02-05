## Overview:

This project investigates decision-making processes by comparing human performance on the Iowa Gambling Task (IGT) to that of AI agents (namely, ChatGPT models) using Bayesian methods.

Human data from Ahn et al. (2014) is compared with ChatGPT IGT data from 2023 (GPT 3.5-turbo-1106) and ChatGPT IGT data from 2026 (GPT 4.1).
The primary model used for this particular Bayesian computational analysis is 'Win-Stay/Lose-Switch' (WSLS).

**The 2023 IGT data, as well as much of this repository's skeleton and code, is obtained from the following repository, by Mina Almasi (@MinaAlmasi) and Anton Drasbæk Schiønning (@drasbaek): https://github.com/MinaAlmasi/ChatGPT-IGT**.

## Repository Structure:

```
├── data/                   # Experimental data (AHN, GPT2023, GPT2025)
├── keys/                   # API keys and credentials (not tracked)
├── models/                 # Computational models used (JAGS)
├── src/                    # Source code for experiments and analysis
│   ├── 1_descriptives/    # Descriptive statistics and exploratory analysis of experimental data
│   ├── 2_estimation/      # Parameter estimation and model fitting
│   ├── 3_simulation/      # Simulated parameter recovery for the chosen model
│   ├── 4_comparison/      # Comparative analysis (LLMs vs humans) with the chosen model
│   ├── data_util.R        # R utilities for data processing
│   ├── gpt_simulate_en.py # GPT simulation script for the IGT experiment
│   └── prepare_real_data.py # Data preprocessing for human participant data
├── utils/                  # Helper functions and utilities
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
└── setup_rjags.sh         # R and JAGS setup script
```

## How to use

- Clone the repo & install `requirements.txt`
- Find or simulate data (`gpt_simulate_en.py` and `prepare_real_data.py`)
- Do the Bayesian analysis steps in order (beginning with `1_descriptives/`)

## WSLS Model used for analysis
The WSLS model captures immediate feedback-driven behavior: repeat choices after wins, switch after losses; it does not include learning dynamics.

It uses three parameters per subject:
prob_stay_win (θ_win): Probability of repeating choice after a win
prob_stay_lose (θ_lose): Probability of repeating choice after a loss
epsilon (ε): Random exploration rate

The hierarchical Bayesian implementation estimates both group-level (population) and individual-level parameters using probit-transformed distributions, enabling robust comparison between human and LLM decision-making patterns.
Learning-based models were not used in this analysis due to the doubt that the GPT LLM models, especially GPT 4.1, were 'learning'.

