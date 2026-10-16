import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontManager

def set_font():
    available_fonts = set(f.name for f in FontManager().ttflist)
    if 'Times New Roman' in available_fonts:
        # use times new roman if available
        plt.rcParams['font.family'] = 'Times New Roman'
        # set font size to 13
        plt.rcParams.update({'font.size': 13})

def calculate_deck_switches(data, window_size=10):
    data = data.copy()
    data['trial'] = data.groupby('subjID').cumcount() + 1
    data['x_last_trial'] = data.groupby('subjID')['x'].shift()
    data = data[data['trial'] != 1]                       # no previous choice on trial 1
    data['deck_switch'] = (data['x'] != data['x_last_trial']).astype(float)

    # smooth within each subject (centered window)
    data['switch_smooth'] = (data.groupby('subjID')['deck_switch']
                             .transform(lambda s: s.rolling(window_size, center=True, min_periods=1).mean()))

    # mean and standard error across subjects for each trial (IT WAS THE GROUP CURVE BEFORE, NOT STANDARD ERROR)
    g = data.groupby('trial')['switch_smooth']
    return pd.DataFrame({'trial': g.mean().index,
                         'mean': g.mean().values,
                         'se': (g.std() / np.sqrt(g.count())).values})

def plot_deck_switch_with_confidence(data1, data2, data3, labels, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#52993C", "#3C5299", "#99523C"]
    alpha = 0.3  # transparency for confidence interval

    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    lines = []
    for i, data in enumerate([data1, data2, data3]):
        ci = 1.96 * data['se']
        line, = ax.plot(data['trial'], data['mean'], label=labels[i], color=colors[i])
        ax.fill_between(data['trial'],
                        np.clip(data['mean'] - ci, 0, 1),
                        np.clip(data['mean'] + ci, 0, 1),
                        color=colors[i], alpha=alpha)
        lines.append(line)

    # switch rate expected under random choice among four decks
    ax.axhline(0.75, linestyle='--', color='grey', linewidth=1)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Proportion of Deck Switches at Trial")
    ax.legend(handles=lines, labels=labels)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def main():
    # define paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data" / "final_data"

    # load both final datasets
    ahn_data = pd.read_csv(data_path / "clean_ahn_hc.csv")
    gpt_data_new = pd.read_csv(data_path / "clean_gpt_NEW_2025.csv")
    gpt_data_old = pd.read_csv(data_path / "clean_gpt_OLD_2023.csv")

    # calculate deck switches for both datasets
    ahn_deck_switch_group = calculate_deck_switches(ahn_data)
    gpt_data_new_switch_group = calculate_deck_switches(gpt_data_new)
    gpt_data_old_switch_group = calculate_deck_switches(gpt_data_old)

    # Set font
    set_font()

    # plot deck switches for both datasets
    plot_deck_switch_with_confidence(ahn_deck_switch_group, gpt_data_old_switch_group, gpt_data_new_switch_group, ['Humans', 'ChatGPT 2023', 'ChatGPT 4.1 (2025)'], save_path=path.parents[2] / "src" / "1_descriptives" / "plots" / "deck_switches.png")


    print(ahn_deck_switch_group)
    print(gpt_data_old_switch_group)
    print(gpt_data_new_switch_group)



if __name__ == "__main__":
    main()