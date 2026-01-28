"""
Functions for simulating ORL data for parameter recovery
"""
import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm

def inv_logit(x):
    return 1 / (1 + np.exp(-x))


def translate_payoff(payoff_df, scale=True):
    """
    Add outcome columns and convert to numpy array
    """
    for letter in ['A', 'B', 'C', 'D']:
        payoff_df[f'{letter}_outcome'] = (
            payoff_df[f'win_{letter}'] - payoff_df[f'loss_{letter}']
        )

    payoff_array = payoff_df[
        ['A_outcome', 'B_outcome', 'C_outcome', 'D_outcome']
    ].to_numpy(dtype=float) #make sure is float (though gives warning?)

    if scale:
        payoff_array /= 100

    return payoff_array


def simulate_ORL(payoff_array, n_trials,
                 a_rew, a_pun, K, theta, omega_f, omega_p):
    """
    Subject-level ORL simulation, made to match the new model
    """

    # ---------- initialize arrays ----------
    x = np.full(n_trials, np.nan, dtype=int)  # choices
    X = np.zeros(n_trials)                     # outcomes

    EV = np.zeros((n_trials, 4))
    EF = np.zeros((n_trials, 4))
    PS = np.ones((n_trials, 4))
    V  = np.zeros((n_trials, 4))
    p  = np.zeros((n_trials, 4))

    deck_counts = {d: 0 for d in range(4)}  # track how many cards drawn per deck

    # ---------- first trial ----------
    p[0, :] = 0.25
    x[0] = np.random.choice(4, p=p[0])
    X[0] = payoff_array[deck_counts[x[0]], x[0]]
    deck_counts[x[0]] += 1

    # ---------- trial loop ----------
    for t in range(1, n_trials):

        # determine sign and learning rates based on last outcome
        last = x[t - 1]
        signX = -1 if X[t - 1] < 0 else 1
        alpha_Ev = a_pun if X[t - 1] < 0 else a_rew
        alpha_Ef_cho = alpha_Ev
        alpha_Ef_not = a_rew if X[t - 1] < 0 else a_pun

        # ---------- update EV, EF, PS, V ----------
        for d in range(4):
            # Expected value
            if d == last:
                EV[t, d] = EV[t - 1, d] + alpha_Ev * (X[t - 1] - EV[t - 1, d])
            else:
                EV[t, d] = EV[t - 1, d]

            # Expected frequency
            if d == last:
                EF[t, d] = EF[t - 1, d] + alpha_Ef_cho * (signX - EF[t - 1, d])
            else:
                EF[t, d] = EF[t - 1, d] + alpha_Ef_not * (-(signX / 3) - EF[t - 1, d])

            # Perseveration
            if d == last:
                PS[t, d] = 1 / (1 + K)
            else:
                PS[t, d] = PS[t - 1, d] / (1 + K)

            # Total value
            V[t, d] = EV[t, d] + EF[t, d] * omega_f + PS[t, d] * omega_p

        # ---------- softmax ----------
        exp_v = np.exp(theta * V[t])
        p[t] = exp_v / exp_v.sum()

        # ---------- handle deck depletion ----------
        available_decks = [d for d in range(4) if deck_counts[d] < payoff_array.shape[0]]

        if not available_decks:
            # all decks depleted, stop simulation
            n_trials_actual = t
            x = x[:n_trials_actual]
            X = X[:n_trials_actual]
            break

        # normalize probabilities over available decks
        prob_choices = p[t, available_decks] / p[t, available_decks].sum()

        # ---------- choose deck ----------
        x[t] = np.random.choice(available_decks, p=prob_choices)
        X[t] = payoff_array[deck_counts[x[t]], x[t]]
        deck_counts[x[t]] += 1

    return {
        "x": x,
        "X": X,
        "n_trials": len(x)
    }


#subject level param recovery
def generate_subject_data(n_iterations, payoff_array, save_path=None):
    """
    Subject-level simulations using SAME priors as hierarchical model (new ver)
    """

    results = []

    for _ in tqdm(range(n_iterations)):

        # ---- group-level priors (fixed) (to see if we can recover them?) ----
        mu_a_rew = -1.5
        mu_a_pun = -2.5
        mu_K = 0
        mu_omega_f = 0
        mu_omega_p = 0

        sigma_a_rew = 1
        sigma_a_pun = 1
        sigma_K = 1
        sigma_omega_f = 1
        sigma_omega_p = 1

        # ---- non-centered draws ----
        z = np.random.normal(0, 1, 5)

        a_rew = inv_logit(mu_a_rew + sigma_a_rew * z[0])
        a_pun = inv_logit(mu_a_pun + sigma_a_pun * z[1])
        K = np.exp(mu_K + sigma_K * z[2])
        omega_f = mu_omega_f + sigma_omega_f * z[3]
        omega_p = mu_omega_p + sigma_omega_p * z[4]

        sim = simulate_ORL(
            payoff_array, 100,
            a_rew, a_pun, K,
            theta=1,
            omega_f=omega_f,
            omega_p=omega_p
        )

        sim.update({
            "a_rew": a_rew,
            "a_pun": a_pun,
            "K": K,
            "omega_f": omega_f,
            "omega_p": omega_p,
            "theta": 1 #fixed for now
        })

        results.append(sim)

    df = pd.DataFrame(results)
    df.index += 1

    if save_path:
        df.to_json(save_path)

    return df


#group level param recovery
def generate_group_data(
    payoff_array,
    n_iterations=20,
    n_subs=48,
    save_path=None
):
    """
    Hierarchical ORL simulation matching JAGS model (new ver)
    """

    results = []

    for _ in tqdm(range(n_iterations)):

        # ---- group-level priors ----
        mu_a_rew = np.random.normal(-1.5, 1)
        mu_a_pun = np.random.normal(-2.5, 1)
        mu_K = np.random.normal(0, 1)

        mu_omega_f = np.random.normal(0, 0.1)
        mu_omega_p = np.random.normal(0, 0.1)

        sigma_a_rew = np.random.uniform(0, 1)
        sigma_a_pun = np.random.uniform(0, 1)
        sigma_K = np.random.uniform(0, 1)
        sigma_omega_f = np.random.uniform(0, 2)
        sigma_omega_p = np.random.uniform(0, 2)

        x = np.zeros((n_subs, 100), dtype=int)
        X = np.zeros((n_subs, 100))

        for s in range(n_subs):

            z = np.random.normal(0, 1, 5)

            a_rew = inv_logit(mu_a_rew + sigma_a_rew * z[0])
            a_pun = inv_logit(mu_a_pun + sigma_a_pun * z[1])
            K = np.exp(mu_K + sigma_K * z[2])
            omega_f = mu_omega_f + sigma_omega_f * z[3]
            omega_p = mu_omega_p + sigma_omega_p * z[4]

            sim = simulate_ORL(
                payoff_array, 100,
                a_rew, a_pun, K,
                theta=1,
                omega_f=omega_f,
                omega_p=omega_p
            )

            x[s] = sim["x"]
            X[s] = sim["X"]

        results.append({
            "x": x,
            "X": X,
            "mu_a_rew": mu_a_rew,
            "mu_a_pun": mu_a_pun,
            "mu_K": mu_K,
            "mu_omega_f": mu_omega_f,
            "mu_omega_p": mu_omega_p,
            "sigma_a_rew": sigma_a_rew,
            "sigma_a_pun": sigma_a_pun,
            "sigma_K": sigma_K,
            "sigma_omega_f": sigma_omega_f,
            "sigma_omega_p": sigma_omega_p
        })

    df = pd.DataFrame(results)
    df.index += 1

    if save_path:
        df.to_json(save_path)

    return df

def main():
    np.random.seed(1234)

    #define paths
    path = pathlib.Path(__file__)
    payoff_path = path.parents[2] / "utils" / "payoff_scheme_3.csv"
    output_dir = path.parents[2] / "src" / "3_simulation" / "recovery" / "simulated_data"
    output_dir.mkdir(exist_ok=True)

    #load payoff
    payoff_df = pd.read_csv(payoff_path)
    payoff_array = translate_payoff(payoff_df)

    #subject-level data
    subject_file = output_dir / "simulated_single_subject_data_fixed_theta.json"
    subject_data = generate_subject_data(
        n_iterations=100,
        payoff_array=payoff_array,
        save_path=subject_file
    )

    #group-level data
    group_file = output_dir / "simulated_group_data_fixed_theta.json"
    group_data = generate_group_data(
        payoff_array=payoff_array,
        n_iterations=100,
        n_subs=30,
        save_path=group_file
    )


    print("Simulation complete. Files saved in:", output_dir)

if __name__ == "__main__":
    main()
