import matplotlib.pyplot as plt


def plot_phase(x, y, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Phase portrait")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
