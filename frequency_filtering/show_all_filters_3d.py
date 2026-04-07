import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from filter_utilities import (
    gaussian_low_pass,
    gaussian_high_pass,
    gaussian_band_pass,
    gaussian_band_reject,
    butterworth_low_pass,
    butterworth_high_pass,
    butterworth_band_pass,
    butterworth_band_reject,
    ideal_low_pass,
    ideal_high_pass,
    ideal_band_pass,
    ideal_band_reject,
    laplacian_filter,
    notch_reject_filter,
)


def plot_filters_3d(filter_specs, shape=(201, 201), cols=3):
    rows = int(np.ceil(len(filter_specs) / cols))
    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    ls = LightSource(azdeg=315, altdeg=45)

    for idx, (title, func, kwargs) in enumerate(filter_specs, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection='3d')
        H = func(shape, **kwargs) if kwargs else func(shape)
        X = np.arange(shape[1])
        Y = np.arange(shape[0])
        X, Y = np.meshgrid(X, Y)
        rgb = ls.shade(H, cmap=plt.get_cmap('viridis'), vert_exag=0.8, blend_mode='soft')
        ax.plot_surface(
            X,
            Y,
            H,
            facecolors=rgb,
            linewidth=0,
            antialiased=True,
            rcount=100,
            ccount=100,
            shade=False,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_facecolor('white')
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.set_box_aspect([1, 1, 0.5])
        ax.view_init(elev=35, azim=-120)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()


def plot_filters_2d(filter_specs, shape=(201, 201), cols=4):
    rows = int(np.ceil(len(filter_specs) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axs = axs.ravel()

    for ax, (title, func, kwargs) in zip(axs, filter_specs):
        H = func(shape, **kwargs) if kwargs else func(shape)
        im = ax.imshow(H, cmap='gray', origin='lower')
        ax.set_title(title)
        ax.axis('off')

    for ax in axs[len(filter_specs):]:
        ax.axis('off')

    fig.colorbar(im, ax=axs.tolist()[:len(filter_specs)], shrink=0.6)
    plt.tight_layout()
    plt.show()


def main():
    filters = [
        ('Gaussian Low Pass', gaussian_low_pass, {'D0': 30}),
        ('Gaussian High Pass', gaussian_high_pass, {'D0': 30}),
        ('Gaussian Band Pass', gaussian_band_pass, {'C0': 90, 'W': 200}),
        ('Gaussian Band Reject', gaussian_band_reject, {'C0': 90, 'W': 200}),
        ('Butterworth Low Pass', butterworth_low_pass, {'D0': 30}),
        ('Butterworth High Pass', butterworth_high_pass, {'D0': 30}),
        ('Butterworth Band Pass', butterworth_band_pass, {'C0': 90, 'W': 200}),
        ('Butterworth Band Reject', butterworth_band_reject, {'C0': 90, 'W': 300}),
        ('Ideal Low Pass', ideal_low_pass, {'radius': 30}),
        ('Ideal High Pass', ideal_high_pass, {'radius': 30}),
        ('Ideal Band Pass', ideal_band_pass, {'C0': 100, 'W': 30}),
        ('Ideal Band Reject', ideal_band_reject, {'C0': 100, 'W': 30}),
        ('Laplacian Filter', laplacian_filter, {}),
        ('Notch Reject Filter', notch_reject_filter, {'d0': 9, 'u_k': 38, 'v_k': 30}),
    ]

    plot_filters_2d(filters, shape=(201, 201), cols=4)
    plot_filters_3d(filters, shape=(201, 201), cols=4)


if __name__ == '__main__':
    main()
