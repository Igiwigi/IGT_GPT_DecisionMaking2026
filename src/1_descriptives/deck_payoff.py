import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontManager


def set_font():
    available_fonts = set(f.name for f in FontManager().ttflist)
    if 'Times New Roman' in available_fonts:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams.update({'font.size': 13})


def calculate_payoffs(data, window_size=10):
    """Per-trial and cumulative net payoff (X = win - loss), mean and SE across subjects."""
    data = data.copy()
    data['trial'] = data.groupby('subjID').cumcount() + 1

    # per-trial payoff, smoothed within each subject (centered window)
    data['payoff_smooth'] = (data.groupby('subjID')['X']
                             .transform(lambda s: s.rolling(window_size, center=True, min_periods=1).mean()))

    # cumulative payoff within each subject (running total)
    data['payoff_cumulative'] = data.groupby('subjID')['X'].cumsum()

    # mean and standard error across subjects for each trial
    g_trial = data.groupby('trial')['payoff_smooth']
    g_cum = data.groupby('trial')['payoff_cumulative']

    return pd.DataFrame({
        'trial': g_trial.mean().index,
        'mean': g_trial.mean().values,
        'se': (g_trial.std() / np.sqrt(g_trial.count())).values,
        'cum_mean': g_cum.mean().values,
        'cum_se': (g_cum.std() / np.sqrt(g_cum.count())).values,
    })


def plot_payoffs_with_confidence(data1, data2, data3, labels, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.3)

    colors = ["#52993C", "#3C5299", "#99523C"]
    alpha = 0.3

    lines = []
    for i, data in enumerate([data1, data2, data3]):
        # top: per-trial net payoff (smoothed)
        ci = 1.96 * data['se']
        line, = ax1.plot(data['trial'], data['mean'], label=labels[i], color=colors[i])
        ax1.fill_between(data['trial'], data['mean'] - ci, data['mean'] + ci,
                         color=colors[i], alpha=alpha)
        lines.append(line)

        # bottom: cumulative net payoff (more important than the top plot)
        cum_ci = 1.96 * data['cum_se']
        ax2.plot(data['trial'], data['cum_mean'], color=colors[i])
        ax2.fill_between(data['trial'], data['cum_mean'] - cum_ci, data['cum_mean'] + cum_ci,
                         color=colors[i], alpha=alpha)

    for ax in (ax1, ax2):
        #zero line
        ax.axhline(0, linestyle='--', color='grey', linewidth=1)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_xlim(0, 100)
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.set_xlabel("Trial")

    ax1.set_ylabel("Net Payoff per Trial (smoothed)")
    ax1.set_title("Net payoff per trial")
    ax2.set_ylabel("Cumulative Net Payoff")
    ax2.set_title("Cumulative net payoff")
    ax1.legend(handles=lines, labels=labels)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def main():
    # define paths & load data
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data" / "final_data"
    
    ahn_data = pd.read_csv(data_path / "clean_ahn_hc.csv")
    gpt_data_new = pd.read_csv(data_path / "clean_gpt_NEW_2025.csv")
    gpt_data_old = pd.read_csv(data_path / "clean_gpt_OLD_2023.csv")

    # calculate payoffs for each dataset
    ahn_payoffs = calculate_payoffs(ahn_data)
    gpt_old_payoffs = calculate_payoffs(gpt_data_old)
    gpt_new_payoffs = calculate_payoffs(gpt_data_new)

    set_font()

    # plot payoffs for the three groups
    plot_payoffs_with_confidence(ahn_payoffs, gpt_old_payoffs, gpt_new_payoffs,
                                 ['Humans', 'ChatGPT 2023', 'ChatGPT 4.1 (2025)'],
                                 save_path=path.parents[2] / "src" / "1_descriptives" / "plots" / "payoffs.png")

    print(ahn_payoffs)
    print(gpt_old_payoffs)
    print(gpt_new_payoffs)


if __name__ == "__main__":
    main()
    