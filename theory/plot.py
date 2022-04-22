"""
Copyright 2021 ServiceNow

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import matplotlib.pyplot as plt
import numpy as np


# function adapted from https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
def boxplot(
    ticks: list, path: str, data_a: np.ndarray, data_b: np.ndarray, xlabel: str, image_name: str, log_scale: bool
):
    """
    Plot boxplots from 2 set of data and save the result
    :param ticks: list of values corresponding to each tick in the x-axis
    :param path: path where to save the generated figure
    :param data_a: first set of data
    :param data_b: second set of data
    :param xlabel: name of the x-axis
    :param image_name: name used to save the figure
    :param log_scale: if True, y axis use a log scale
    """

    def set_box_color(bp, color, fillcolor):
        """Changes the color of the boxes"""
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)

        for patch in bp["boxes"]:
            patch.set(facecolor=fillcolor)

    # Plot the boxplots
    ax = plt.figure().add_subplot(111)

    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a))) * 2.0 - 0.4, sym="", widths=0.6, patch_artist=True)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym="", widths=0.6, patch_artist=True)
    set_box_color(bpl, "#D7191C", "#FFDABD")  # colors are from http://colorbrewer2.org/
    set_box_color(bpr, "#2C7BB6", "#C2E4FF")

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c="#D7191C", label="MEC")
    plt.plot([], c="#2C7BB6", label="t-MEC")
    plt.legend()

    # Set the x and y axes
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)

    plt.xlabel(xlabel)
    if log_scale:
        plt.yscale("log")

    plt.ylabel("Size of equivalence class")
    plt.tight_layout()

    # change the y axis to display 1 as the
    # first entry
    yticks = ax.get_yticks().tolist()
    yticks[1] = 1
    yticks = [int(i) for i in yticks]
    ax.set_yticklabels(yticks)

    # save the figure
    plt.savefig(f"{path}/{image_name}.png")
