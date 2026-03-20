import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

def plot_walsh():
    n = 8
    H = hadamard(n)
    fig, axes = plt.subplots(n, 1, figsize=(8, 10))
    for i in range(n):
        axes[i].step(range(n), H[i], where='post', color='black', lw=2)
        axes[i].set_ylim(-1.5, 1.5)
        axes[i].set_ylabel(f'wal({i}, t)', rotation=0, labelpad=30)
        axes[i].set_xticks([])
        axes[i].set_yticks([-1, 1])
    plt.suptitle("Ուոլշի առաջին 8 բազիսային ֆունկցիաները (Հադամարի կարգավորմամբ)")
    plt.tight_layout()
    plt.show()

plot_walsh()
