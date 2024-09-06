#!/usr/bin/env python3
"""plot a stacked bar graph"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """bar graph: """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    fig, ax = plt.subplots()
    person = ["Farrah", "Fred", "Felicia"]
    color = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fru = ['apples', 'bananas', 'oranges', 'peaches']
    bottom = np.zeros(len(person))
    for i, row in enumerate(fruit):
        ax.bar(person, row, bottom=bottom,
               color=color[i], label=fru[i], width=0.5)
        bottom += row

    ax.set_ylabel('Quantity of Fruit')
    ax.set_title('Number of Fruit per Person')
    ax.set_ylim([0, 80])
    ax.set_yticks(np.arange(0, 80, 10))
    ax.legend()
    plt.show()
