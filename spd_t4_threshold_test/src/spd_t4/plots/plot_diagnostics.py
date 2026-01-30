import matplotlib.pyplot as plt


def plot_early_warning(time, rolling_var, rolling_ac1, out_path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(time, rolling_var, label="rolling var")
    ax[0].set_ylabel("variance")
    ax[0].legend()

    ax[1].plot(time, rolling_ac1, label="lag-1 autocorr")
    ax[1].set_ylabel("ac1")
    ax[1].set_xlabel("time")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
