import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from common.utils import center_crop_image, random_crop, uniquify
def save_tsne_plots(env, filename, cropped_img_size, actor):
    """
    Save t-SNE plots of the states of the environment.
    """
    for _ in range(5):
        obs = env.reset()
        done = False
        while not done:
            states = center_crop_image (obs, cropped_img_size)
            state_embed = actor.encode(states)
            actions = actor.policy(states)
            next_obs, rewards, dones, _ = env.step(actions)
            obs = next_obs

        states = np.array(states)
        states = states.reshape(states.shape[0], -1)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(states)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.savefig(filename)
        plt.close()

    # get the states
    states = env.get_states()
    # get the embedding
    embedding = get_embedding(states)
    # get the labels
    labels = env.get_labels()
    # get the colors
    colors = env.get_colors()
    # get the names
    names = env.get_names()
    # plot
    plot_tsne(embedding, labels, colors, names, filename)``