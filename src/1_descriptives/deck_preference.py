import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.stats import binom
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontManager

def set_font():
    available_fonts = set(f.name for f in FontManager().ttflist)
    if 'Times New Roman' in available_fonts:
        # use times new roman if available
        plt.rcParams['font.family'] = 'Times New Roman'
        # set font size to 13
        plt.rcParams.update({'font.size': 12})

def calculate_deck_preferences(data, window_size=10):
    data = data.copy()
    data['trial'] = data.groupby('subjID').cumcount() + 1
    data['x'] = data['x'].replace([1, 2, 3, 4], ['A', 'B', 'C', 'D'])

    # one indicator column per deck (1 if chosen on that trial)
    decks = ['A', 'B', 'C', 'D']
    dummies = pd.get_dummies(data['x']).reindex(columns=decks, fill_value=0).astype(float)
    dummies['subjID'] = data['subjID']
    dummies['trial'] = data['trial']

    # smooth within each subject (centered window)
    smoothed = (dummies.groupby('subjID')[decks]
                .transform(lambda s: s.rolling(window_size, center=True, min_periods=1).mean()))
    smoothed['trial'] = dummies['trial']

    # mean and standard error across subjects for each trial (IT WAS THE GROUP CURVE BEFORE, NOT STANDARD ERROR)
    g = smoothed.groupby('trial')[decks]
    mean = g.mean()
    se = g.std() / np.sqrt(g.count())
    return mean, se


def plot_deck_preferences(data1, data2, data3, labels, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5, top=0.90)

    colors = ["#F59B8C", "#B43622", "#A1E780", "#3CA50A"]
    decks = ['A', 'B', 'C', 'D']

    for i, (ax, (mean, se)) in enumerate(zip(axes, [data1, data2, data3])):
        trials = mean.index
        for j, deck in enumerate(decks):
            ci = 1.96 * se[deck]
            ax.plot(trials, mean[deck], color=colors[j])
            ax.fill_between(trials,
                            np.clip(mean[deck] - ci, 0, 1),
                            np.clip(mean[deck] + ci, 0, 1),
                            color=colors[j], alpha=0.1)

        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.set_title(labels[i], pad=10)
        ax.set_ylabel('Choice Proportions')
        ax.set_xlabel('Trial number')
        ax.set_xlim(0, 100)

    axes[0].set_ylim(0, 0.5)
    axes[1].set_ylim(0, 0.5)
    axes[2].set_ylim(0, 0.9)

    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    fig.legend(custom_lines, ['Deck A', 'Deck B', 'Deck C', 'Deck D'],
               loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def main():
    # define paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data" / "final_data"

    # load both final datasets
    ahn_data = pd.read_csv(data_path / "clean_ahn_hc.csv")
    gpt_data_old = pd.read_csv(data_path / "clean_gpt_OLD_2023.csv")
    gpt_data_new = pd.read_csv(data_path / "clean_gpt_NEW_2025.csv")


    # calculate deck preferences
    ahn_deck_preferences = calculate_deck_preferences(ahn_data)
    gpt_deck_preferences_old = calculate_deck_preferences(gpt_data_old)
    gpt_deck_preferences_new = calculate_deck_preferences(gpt_data_new)

    # set font
    set_font()

    # plot deck preferences
    plot_deck_preferences(ahn_deck_preferences, gpt_deck_preferences_old, gpt_deck_preferences_new, labels=['Humans', 'ChatGPT 2023', 'ChatGPT 4.1 (2025)'], save_path=path.parents[2] / "src" / "1_descriptives" / "plots" / "deck_preferences.png")





if __name__ == "__main__":
    main()