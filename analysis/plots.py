import matplotlib.pyplot as plt


def plot_returns(returns, path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Courbe de retour")
    plt.grid(True)
    if path:
        plt.savefig(path)
    else:
        plt.show()
