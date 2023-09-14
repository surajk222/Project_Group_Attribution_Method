import matplotlib.pyplot as plt
import numpy as np


def _visualize_log_odds(
    log_odds: np.ndarray,
    mean: np.ndarray,
    max: np.ndarray,
    min: np.ndarray,
    random_references_mean: np.ndarray,
    apply_log: bool,
) -> None:
    """
    Serves the IntegratedGradientEvaluator class for visualization.
    """
    x = np.linspace(0, 1, 16)
    ax = plt.plot(x, mean, label="Mean", linestyle="-", color="black")
    plt.plot(x, max, label="Max", linestyle="--", color="gray")
    plt.plot(x, min, label="Min", linestyle=":", color="gray")
    plt.plot(x, random_references_mean, label="Mean of Random Reference")

    plt.xlabel("Anteil der maskierten Features")

    if apply_log:
        plt.ylabel("Log Odds Ratio")
    else: 
        plt.ylabel("Modellausgabe")

    if apply_log:
        plt.title("Log Odds von Integrated Gradients")

    else: 
        plt.title("Certainty Kurve von Integrated Gradients")

    plt.legend()
    plt.tight_layout()
    plt.savefig("log_odds_ig.eps", format="eps")
    # plt.show()

def _visualize_completeness_deltas_comparison(
        completeness_deltas_list : np.ndarray,
        n_steps_arr: list[int]
    ) -> None:

    plt.plot(n_steps_arr, completeness_deltas_list)
    plt.xlabel("Anzahl der Approximationsschritte m")
    plt.ylabel("Delta")

    plt.ticklabel_format(style="sci",axis="y",scilimits=(0,0))

    plt.tight_layout()
    plt.show()
