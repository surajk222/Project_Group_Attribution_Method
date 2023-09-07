import matplotlib.pyplot as plt
import numpy as np


def _visualize_log_odds(
        log_odds: np.ndarray,
        mean: np.ndarray,
        max: np.ndarray,
        min: np.ndarray
) -> None:
    """
    Serves the IntegratedGradientEvaluator class for visualization.
    """
    x = np.linspace(0,1,16)
    ax = plt.plot(x, mean, label="Mean",linestyle="-", color="black")
    plt.plot(x,max,label="Max",linestyle="--", color="gray")
    plt.plot(x,min,label="Min", linestyle=":", color="gray")

    plt.xlabel("Anteil der maskierten Features")
    plt.ylabel("Log Odds Ratio")

    plt.title("Log Odds von Integrated Gradients")

    plt.legend()
    plt.tight_layout()
    plt.savefig('log_odds_ig.eps', format='eps')
    #plt.show()


  