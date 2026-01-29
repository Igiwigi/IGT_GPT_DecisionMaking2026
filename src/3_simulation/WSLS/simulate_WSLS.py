import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm
from scipy.stats import norm

def phi(x):
    """Probit inverse: transform from real line to [0,1]"""
    return norm.cdf(x)

def translate_payoff(payoff_df, scale=True):
    """
    Convert payoff_df to outcome array [trials x 4 decks].
    """
    for letter in ['A', 'B', 'C', 'D']:
        payoff_df[f'{letter}_outcome'] = payoff_df[f'win_{letter}'] - payoff_df[f'loss_{letter}']
    
    payoff_array = payoff_df[['A_outcome', 'B_outcome', 'C_outcome', 'D_outcome']].to_numpy(dtype=float)
    if scale:
        payoff_array /= 100
    return payoff_array

def simulate_winstay_loseepsilon(payoff_array, n_trials, prob_stay_win, prob_stay_lose, epsilon):
    """
    Simulate a single subject using win-stay / lose-stay + epsilon.
    """

    x = np.full(n_trials, np.nan, dtype=int)
    X = np.zeros(n_trials)
    p = np.zeros((n_trials, 4))

    # First trial: equal probabilities
    p[0] = np.ones(4) / 4
    x[0] = np.random.choice(4, p=p[0])
    X[0] = payoff_array[0, x[0]]

    deck_counts = {d: 1 if d == x[0] else 0 for d in range(4)}

    for t in range(1, n_trials):
        last_choice = x[t-1]
        last_outcome = X[t-1]

        # determine if previous trial was a win
        is_win = last_outcome >= 0
        stay_prob = prob_stay_win if is_win else prob_stay_lose

        # compute raw stay probabilities
        p_raw = np.full(4, (1 - stay_prob) / 3)
        p_raw[last_choice] = stay_prob

        # add epsilon (random exploration)
        p[t] = (1 - epsilon) * p_raw + epsilon * 0.25

        # handle deck depletion
        available_decks = [d for d in range(4) if deck_counts[d] < payoff_array.shape[0]]
        if not available_decks:
            # stop if all decks depleted
            x = x[:t]
            X = X[:t]
            break

        # normalize over available decks
        p_available = p[t, available_decks] / p[t, available_decks].sum()
        x[t] = np.random.choice(available_decks, p=p_available)
        X[t] = payoff_array[deck_counts[x[t]], x[t]]
        deck_counts[x[t]] += 1

    return {"x": x, "X": X, "n_trials": len(x)}

def generate_subject_data_winlose(n_iterations, payoff_array, save_path=None):
    """
    Subject-level simulations using probit-transformed priors.
    """
    results = []

    for _ in tqdm(range(n_iterations)):
        # group-level hyperpriors
        mu_prob_stay_win_raw = 0
        mu_prob_stay_lose_raw = 0
        mu_epsilon_raw = 0

        lambda_prob_stay_win = 1
        lambda_prob_stay_lose = 1
        lambda_epsilon = 1

        # sample subject-level parameters (probit transformed)
        z = np.random.normal(0, 1, 3)
        prob_stay_win = phi(mu_prob_stay_win_raw + z[0] / np.sqrt(lambda_prob_stay_win))
        prob_stay_lose = phi(mu_prob_stay_lose_raw + z[1] / np.sqrt(lambda_prob_stay_lose))
        epsilon = phi(mu_epsilon_raw + z[2] / np.sqrt(lambda_epsilon))

        sim = simulate_winstay_loseepsilon(
            payoff_array,
            n_trials=100,
            prob_stay_win=prob_stay_win,
            prob_stay_lose=prob_stay_lose,
            epsilon=epsilon
        )

        sim.update({
            "prob_stay_win": prob_stay_win,
            "prob_stay_lose": prob_stay_lose,
            "epsilon": epsilon
        })

        results.append(sim)

    df = pd.DataFrame(results)
    df.index += 1

    if save_path:
        df.to_json(save_path)

    return df

def generate_group_data_winlose(payoff_array, n_iterations=20, n_subs=48, save_path=None):
    """
    Hierarchical group-level simulations using probit-transformed priors.
    """
    results = []

    for _ in tqdm(range(n_iterations)):
        # Sample group-level hyperpriors for this iteration
        mu_prob_stay_win_raw = np.random.normal(0, 1)
        mu_prob_stay_lose_raw = np.random.normal(0, 1)
        mu_epsilon_raw = np.random.normal(0, 1)

        lambda_prob_stay_win = np.random.uniform(0.01, 2)
        lambda_prob_stay_lose = np.random.uniform(0.01, 2)
        lambda_epsilon = np.random.uniform(0.01, 2)

        x = np.zeros((n_subs, 100), dtype=int)
        X = np.zeros((n_subs, 100))

        prob_stay_win_sub = np.zeros(n_subs)
        prob_stay_lose_sub = np.zeros(n_subs)
        epsilon_sub = np.zeros(n_subs)

        for s in range(n_subs):
            # sample subject-level parameters
            z = np.random.normal(0, 1, 3)
            prob_stay_win = phi(mu_prob_stay_win_raw + z[0] / np.sqrt(lambda_prob_stay_win))
            prob_stay_lose = phi(mu_prob_stay_lose_raw + z[1] / np.sqrt(lambda_prob_stay_lose))
            epsilon = phi(mu_epsilon_raw + z[2] / np.sqrt(lambda_epsilon))

            prob_stay_win_sub[s] = prob_stay_win
            prob_stay_lose_sub[s] = prob_stay_lose
            epsilon_sub[s] = epsilon

            sim = simulate_winstay_loseepsilon(
                payoff_array,
                n_trials=100,
                prob_stay_win=prob_stay_win,
                prob_stay_lose=prob_stay_lose,
                epsilon=epsilon
            )

            x[s, :sim["n_trials"]] = sim["x"]
            X[s, :sim["n_trials"]] = sim["X"]

        results.append({
            "x": x,
            "X": X,
            "mu_prob_stay_win_raw": mu_prob_stay_win_raw,
            "mu_prob_stay_lose_raw": mu_prob_stay_lose_raw,
            "mu_epsilon_raw": mu_epsilon_raw,
            "lambda_prob_stay_win": lambda_prob_stay_win,
            "lambda_prob_stay_lose": lambda_prob_stay_lose,
            "lambda_epsilon": lambda_epsilon,
            "prob_stay_win_sub": prob_stay_win_sub,
            "prob_stay_lose_sub": prob_stay_lose_sub,
            "epsilon_sub": epsilon_sub
        })

    df = pd.DataFrame(results)
    df.index += 1

    if save_path:
        df.to_json(save_path)

    return df


def main():
    np.random.seed(1234)

    path = pathlib.Path(__file__)
    payoff_path = path.parents[3] / "utils" / "payoff_scheme_3.csv"
    output_dir = path.parents[3] / "src" / "3_simulation" / "WSLS" / "recovery" / "simulated_data"
    output_dir.mkdir(exist_ok=True, parents=True)

    payoff_df = pd.read_csv(payoff_path)
    payoff_array = translate_payoff(payoff_df)

    # Subject-level simulation
    subject_file = output_dir / "simulated_single_subject_winlose.json"
    subject_data = generate_subject_data_winlose(
        n_iterations=100,
        payoff_array=payoff_array,
        save_path=subject_file
    )
    
    # Group-level simulation
    group_file = output_dir / "simulated_group_winlose.json"
    group_data = generate_group_data_winlose(
        payoff_array,
        n_iterations=100,
        n_subs=30,
        save_path=group_file
)

    print("Simulation complete. Files saved in:", output_dir)

if __name__ == "__main__":
    main()
