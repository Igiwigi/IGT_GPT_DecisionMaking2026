import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm
from scipy.stats import norm

# ============================================================================
# WIN-STAY / LOSE-SWITCH (WSLS) SIMULATIONS FOR THE IGT - NO EPSILON
# Matches the JAGS models (e.g. WSLS_hier_grp_no_epsilon.txt):
#   - win  = previous net outcome >= 0
#   - P(stay) after a win  = prob_stay_win
#   - P(stay) after a loss = 1 - prob_switch_lose
#   - the other three decks split the remaining probability equally
#   - each deck has 60 cards; empty decks get probability 0 and the rest is renormalized
#     (the JAGS model must do the same, via an availability array)
#   - parameters live on the probit scale; lambda is a PRECISION (sd = 1/sqrt(lambda))
# ============================================================================

N_TRIALS = 100


def phi(x):
    """Probit inverse: transform from real line to [0,1]"""
    return norm.cdf(x)


def translate_payoff(payoff_df, scale=True):
    """
    Convert payoff_df to outcome array [trials x 4 decks] of NET outcomes (win - loss).
    """
    for letter in ['A', 'B', 'C', 'D']:
        payoff_df[f'{letter}_outcome'] = payoff_df[f'win_{letter}'] - payoff_df[f'loss_{letter}']

    payoff_array = payoff_df[['A_outcome', 'B_outcome', 'C_outcome', 'D_outcome']].to_numpy(dtype=float)
    if scale:
        payoff_array /= 100
    return payoff_array


def sample_subject_params(mu_stay_win_raw, mu_switch_lose_raw, lambda_stay_win, lambda_switch_lose):
    """
    Draw one subject's parameters from the group distribution (probit scale),
    exactly like JAGS: raw ~ dnorm(mu, lambda), prob = phi(raw).
    """
    z = np.random.normal(0, 1, 2)
    prob_stay_win = phi(mu_stay_win_raw + z[0] / np.sqrt(lambda_stay_win))
    prob_switch_lose = phi(mu_switch_lose_raw + z[1] / np.sqrt(lambda_switch_lose))
    return prob_stay_win, prob_switch_lose


def simulate_wsls(payoff_array, n_trials, prob_stay_win, prob_switch_lose):
    """
    Simulate one subject playing the IGT with win-stay / lose-switch (no epsilon).
    """
    n_cards = payoff_array.shape[0]          # cards per deck (60)
    assert 4 * n_cards >= n_trials, "not enough cards in total for n_trials"

    x = np.zeros(n_trials, dtype=int)       # deck chosen (0-3)
    X = np.zeros(n_trials)                  # net outcome received
    deck_counts = np.zeros(4, dtype=int)    # how many cards drawn from each deck so far

    # Trial 1: no previous choice, so pick uniformly (not modelled in JAGS)
    x[0] = np.random.choice(4)
    X[0] = payoff_array[0, x[0]]
    deck_counts[x[0]] += 1

    for t in range(1, n_trials):
        # Was the previous trial a win?
        is_win = X[t-1] >= 0

        # Probability of switching away from the previous deck
        switch_prob = 1 - prob_stay_win if is_win else prob_switch_lose

        # Switching spreads equally over the 3 other decks; the rest stays
        p = np.full(4, switch_prob / 3)
        p[x[t-1]] = 1 - switch_prob

        # Empty decks can't be chosen: zero them out and renormalize over the rest.
        # If the previous deck is empty, the subject is forced to switch.
        available = deck_counts < n_cards
        p = p * available
        if p.sum() == 0:
            # e.g. stay prob 1 on an empty deck: fall back to uniform over available decks
            p = available.astype(float)
        p = p / p.sum()

        # Choose, then draw the next card from that deck's own sequence
        x[t] = np.random.choice(4, p=p)
        X[t] = payoff_array[deck_counts[x[t]], x[t]]
        deck_counts[x[t]] += 1

    return {"x": x, "X": X, "n_trials": n_trials}


def generate_subject_data_winlose(n_iterations, payoff_array, save_path=None):
    """
    Single-subject simulations: each iteration is one independent subject.
    mu = 0 and lambda = 1 on the probit scale, so each probability is ~Uniform(0, 1).
    """
    results = []

    for _ in tqdm(range(n_iterations), desc="Single-subject simulations"):
        prob_stay_win, prob_switch_lose = sample_subject_params(0, 0, 1, 1)

        sim = simulate_wsls(payoff_array, N_TRIALS, prob_stay_win, prob_switch_lose)
        sim.update({
            "prob_stay_win": prob_stay_win,
            "prob_switch_lose": prob_switch_lose
        })
        results.append(sim)

    df = pd.DataFrame(results)
    df.index += 1

    if save_path:
        df.to_json(save_path)

    return df


def generate_group_data_winlose(payoff_array, n_iterations=20, n_subs=48, save_path=None):
    """
    Hierarchical single-group simulations: each iteration draws new group-level
    mu and lambda from the JAGS priors, then n_subs subjects from that group.
    """
    results = []

    for _ in tqdm(range(n_iterations), desc="Group simulations"):
        # Group-level hyperparameters, drawn from the same priors as JAGS
        mu_prob_stay_win_raw = np.random.normal(0, 1)              # dnorm(0, 1)
        mu_prob_switch_lose_raw = np.random.normal(0, 1)           # dnorm(0, 1)
        lambda_prob_stay_win = np.random.gamma(shape=1, scale=2)     # dgamma(1, 0.5) (rate 0.5 = scale 2)
        lambda_prob_switch_lose = np.random.gamma(shape=1, scale=2)  # dgamma(1, 0.5)

        x = np.zeros((n_subs, N_TRIALS), dtype=int)
        X = np.zeros((n_subs, N_TRIALS))
        prob_stay_win_sub = np.zeros(n_subs)
        prob_switch_lose_sub = np.zeros(n_subs)

        for s in range(n_subs):
            prob_stay_win, prob_switch_lose = sample_subject_params(
                mu_prob_stay_win_raw, mu_prob_switch_lose_raw,
                lambda_prob_stay_win, lambda_prob_switch_lose
            )
            prob_stay_win_sub[s] = prob_stay_win
            prob_switch_lose_sub[s] = prob_switch_lose

            sim = simulate_wsls(payoff_array, N_TRIALS, prob_stay_win, prob_switch_lose)
            x[s, :] = sim["x"]
            X[s, :] = sim["X"]

        results.append({
            "x": x,
            "X": X,
            "mu_prob_stay_win_raw": mu_prob_stay_win_raw,
            "mu_prob_switch_lose_raw": mu_prob_switch_lose_raw,
            "lambda_prob_stay_win": lambda_prob_stay_win,
            "lambda_prob_switch_lose": lambda_prob_switch_lose,
            "prob_stay_win_sub": prob_stay_win_sub,
            "prob_switch_lose_sub": prob_switch_lose_sub
        })

    df = pd.DataFrame(results)
    df.index += 1

    if save_path:
        df.to_json(save_path)

    return df


def generate_group_comparison_data_winlose(payoff_array, n_iterations=30, n_subs_per_group=30, save_path=None):
    """
    Group comparison simulations with 3 groups: human, gpt2023, gpt2025.
    True group parameters are FIXED across iterations; only the subjects and their choices change.
    Key names match what the R recovery script reads (prob_switch_lose_raw / prob_switch_lose).
    """
    
    #these are all based on the results from estimation (taking means, basically)
    true_params = {
        'human': {
            'mu': {
                'prob_stay_win_raw': norm.ppf(0.30),     # humans dont really stay after a win that often
                'prob_switch_lose_raw': norm.ppf(0.85),  # switches a lot after losses
            },
            'lambda': {
                'prob_stay_win': 2.6, #moderate variance
                'prob_switch_lose': 1.8, #some variation
            }
        },
        'gpt2023': {
            'mu': {
                'prob_stay_win_raw': norm.ppf(0.44),     # more than chance
                'prob_switch_lose_raw': norm.ppf(0.73),  # mostly random
            },
            'lambda': {
                'prob_stay_win': 0.6, #large differences
                'prob_switch_lose': 1.7, #moderate differences
            }
        },
        'gpt2025': {
            'mu': {
                'prob_stay_win_raw': norm.ppf(0.93),     # highest
                'prob_switch_lose_raw': norm.ppf(0.88),  # switches most after losses
            },
            'lambda': {
                'prob_stay_win': 1.1, #some variation
                'prob_switch_lose': 4.1, #consistent
            }
        }
    }

    group_names = ['human', 'gpt2023', 'gpt2025']
    results = []

    for iteration in tqdm(range(n_iterations), desc="Group comparison simulations"):
        total_subs = n_subs_per_group * len(group_names)
        x = np.zeros((total_subs, N_TRIALS), dtype=int)
        X = np.zeros((total_subs, N_TRIALS))
        prob_stay_win_all = np.zeros(total_subs)
        prob_switch_lose_all = np.zeros(total_subs)
        group_labels = []

        sub_idx = 0
        for group_name in group_names:
            mu = true_params[group_name]['mu']
            lam = true_params[group_name]['lambda']

            for s in range(n_subs_per_group):
                prob_stay_win, prob_switch_lose = sample_subject_params(
                    mu['prob_stay_win_raw'], mu['prob_switch_lose_raw'],
                    lam['prob_stay_win'], lam['prob_switch_lose']
                )
                prob_stay_win_all[sub_idx] = prob_stay_win
                prob_switch_lose_all[sub_idx] = prob_switch_lose
                group_labels.append(group_name)

                sim = simulate_wsls(payoff_array, N_TRIALS, prob_stay_win, prob_switch_lose)
                x[sub_idx, :] = sim["x"]
                X[sub_idx, :] = sim["X"]

                sub_idx += 1

        results.append({
            'iteration': iteration + 1,
            'x': x,
            'X': X,
            'group': group_labels,
            'prob_stay_win': prob_stay_win_all,
            'prob_switch_lose': prob_switch_lose_all,
            'true_params': true_params
        })

    df = pd.DataFrame(results)

    if save_path:
        df.to_json(save_path)

    return df


def main():
    np.random.seed(1234)

    path = pathlib.Path(__file__)
    payoff_path = path.parents[2] / "utils" / "payoff_scheme_3.csv"
    output_dir = path.parents[2] / "src" / "3_simulation" / "recovery" / "simulated_data"
    output_dir.mkdir(exist_ok=True, parents=True)

    payoff_df = pd.read_csv(payoff_path)
    payoff_array = translate_payoff(payoff_df)

    # Single-subject simulation
    generate_subject_data_winlose(
        n_iterations=100,
        payoff_array=payoff_array,
        save_path=output_dir / "simulated_single_subject_winlose.json"
    )

    # Single-group hierarchical simulation
    generate_group_data_winlose(
        payoff_array,
        n_iterations=100,
        n_subs=30,
        save_path=output_dir / "simulated_group_winlose.json"
    )

    # Group comparison simulation (to get group-level differences)
    generate_group_comparison_data_winlose(
        payoff_array,
        n_iterations=30,
        n_subs_per_group=30,
        save_path=output_dir / "simulated_group_comparison_winlose.json"
    )

    print("Simulation complete. Files saved in:", output_dir)


if __name__ == "__main__":
    main()