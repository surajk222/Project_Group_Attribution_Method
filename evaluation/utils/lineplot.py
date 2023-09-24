import matplotlib.pyplot as plt
import numpy as np

# visualizing changes in attribution scores with changing hyperparameters

def visualize_scores(
        attribution_scores,
        change,
        title,
        hyperparameter,
        xlabel,
        start,
        end,
        step_size
):
    
    fig, ax = plt.subplots()
    steps = np.array([])
    for i in np.arange(start, end, step_size):
        steps = np.append(steps, i)
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Scores / \nSumme der Differenzen', fontsize=16)
    lines = ax.plot(steps, attribution_scores, linestyle=':', color='grey')
    ax.plot(steps, change, label='Summe der Differenzen', color='blue')
    ax.legend(fontsize=14, bbox_to_anchor =(0.5,-0.2), loc='lower center')

    # plt.tight_layout()
    # plt.setp(lines[1:], label="_")
    plt.savefig("./figures/"+hyperparameter+"2.eps", format="eps")

    plt.show()
