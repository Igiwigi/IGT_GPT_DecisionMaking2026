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

def calculate_deck_preferences(data):
    # create a new column that indicates the trial number
    data['trial'] = data.groupby('subjID').cumcount() + 1

    # replace 1,2,3,4 with A,B,C,D for 'x' column
    data['x'] = data['x'].replace([1,2,3,4], ['A','B','C','D'])

    grouped = data.groupby(['trial', 'x']).size().unstack(fill_value=0)
    # Calculate the proportion of each 'x' value in each trial
    proportions = grouped.div(grouped.sum(axis=1), axis=0)

    # Reset the index to have 'trial' as a column
    proportions.reset_index(inplace=True)

    return proportions


def plot_deck_preferences(data1, data2, data3, labels, window_size=10, save_path=None):
    # Create a plot with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # add space between subplots
    fig.subplots_adjust(hspace=0.5, top=0.90)

    # Define colors for all four decks
    colors = ["#F59B8C", "#B43622", "#A1E780", "#3CA50A"]

    # loop over both subplots and datasets
    for i, (ax, data) in enumerate(zip([ax1, ax2, ax3], [data1, data2, data3])):
        # Apply smoothing
        smooth_data = data.iloc[:, 1:].rolling(window=window_size, min_periods=1).mean()
        std_error = data.iloc[:, 1:].rolling(window=window_size, min_periods=1).std()
        
        # Calculate confidence interval (95%)
        ci = 1.96 * std_error / np.sqrt(window_size)

        alpha = 0.1

        # Plotting line and shaded confidence interval on the correct axis
        for j, deck in enumerate(smooth_data.columns):
            ax.plot(data['trial'], smooth_data[deck], label=deck, color=colors[j])
            lower_bound = np.maximum(smooth_data[deck] - ci[deck], 0)
            upper_bound = np.minimum(smooth_data[deck] + ci[deck], 1)
            ax.fill_between(data['trial'], lower_bound, upper_bound, color=colors[j], alpha=alpha)
        
        # set x-axis ticks to every 10th trial
        ax.set_xticks(np.arange(0, 101, 10))

        # set y-axis ticks to every 0.1
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        # add very light grid
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        # set title
        ax.set_title(labels[i], pad=10)

    # add custom legend
    custom_lines = [Line2D([0], [0], color=colors[0], lw=2),
                    Line2D([0], [0], color=colors[1], lw=2),
                    Line2D([0], [0], color=colors[2], lw=2),
                    Line2D([0], [0], color=colors[3], lw=2)]

    # add legend
    fig.legend(custom_lines, ['Deck A', 'Deck B', 'Deck C', 'Deck D'], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)

    # add y-axis label
    ax1.set_ylabel('Choice Proportions')
    ax2.set_ylabel('Choice Proportions')
    ax3.set_ylabel('Choice Proportions')

    # add x-axis label
    ax1.set_xlabel('Trial number')
    ax2.set_xlabel('Trial number')
    ax3.set_xlabel('Trial number')

    # expand y axis
    ax1.set_ylim(0, 0.5)
    ax2.set_ylim(0, 0.5)
    ax3.set_ylim(0, 0.9)
    
    #limit x axis
    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)
    ax3.set_xlim(0, 100)

    # save the plot
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