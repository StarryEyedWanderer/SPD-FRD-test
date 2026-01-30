import matplotlib.pyplot as plt


def plot_eps_theta(time, radius, eps, theta, t_star, t_delta, out_path):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(time, radius, label="radius")
    if t_star is not None:
        ax[0].axvline(time[t_star], color="k", linestyle="--", label="t*")
    if t_delta is not None:
        ax[0].axvline(time[t_delta], color="r", linestyle=":", label="tΔ")
    ax[0].set_ylabel("r(t)")
    ax[0].legend()

    ax[1].plot(time, eps, label="epsilon")
    ax[1].axhline(theta, color="r", linestyle="--", label="theta")
    ax[1].set_ylabel("epsilon")
    ax[1].legend()

    ax[2].plot(time, radius, label="radius")
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("r(t)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
