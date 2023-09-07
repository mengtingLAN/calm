import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
print(os.getcwd())

def main():
    print(os.getcwd())
    data_path = '../output/dynamic_log/c6930a92_dynamic_sample_result.csv'
    data = np.array(pd.read_csv(data_path))
    _, avg_reward, sample_times = np.split(data, 3, axis=1)
    print(np.argsort(avg_reward)[-1, :20])
    shape = (2, 41)
    # shape = (8, 14)
    avg_reward = avg_reward.reshape(avg_reward.shape[0], *shape)
    sample_times = sample_times.reshape(sample_times.shape[0], *shape)
    frames = avg_reward.shape[0] if avg_reward.shape[0] < 100 else 100
    def animate(i):
        ax[0, 0].cla()
        ax[1, 0].cla()
        ax[0, 0].title.set_text('Sample Frequency')
        ax[1, 0].title.set_text('Average Reward per step')
        sns.heatmap(ax=ax[0, 0], data=sample_times[-frames+i], cmap="coolwarm", cbar_ax=ax[0, 1])
        sns.heatmap(ax=ax[1, 0], data=avg_reward[-frames+i], cmap="coolwarm", cbar_ax=ax[1, 1], vmin=0, vmax=1.0)

    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, ax = plt.subplots(2, 2, gridspec_kw=grid_kws, figsize=(10, 8))
    print("avg performance latest: ", np.mean(avg_reward[-1]))
    print("avg performance max: ", max(np.mean(avg_reward.reshape(avg_reward.shape[0], -1), axis=1)))
    ani = FuncAnimation(fig=fig, func=animate, frames=frames, interval=1, repeat=False)
    plt.show()


if __name__ == '__main__':
    main()
