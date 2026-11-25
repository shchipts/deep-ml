import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_clusters(specs, X, n_cols=2, size=[8.5, 2.5]):
    n_rows = math.ceil(len(specs) / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(size[0], size[1] * n_rows), sharex=True, sharey=True
    )

    axes = np.array(axes).flatten()

    for j in range(len(specs), len(axes)):
        axes[j].set_visible(False)

    cmap = plt.get_cmap('tab10')

    for i, spec in enumerate(specs):
        fit = spec.get('fit', None)
        n_classes = spec.get('n', None)
        label = spec['label']
        predict = spec.get('predict', lambda m, X_: m.predict(X_))

        clf = (
            (fit(X, n_classes) if n_classes is not None else fit(X))
            if fit is not None
            else None
        )
        Y = predict(clf, X)
        if n_classes is None:
            n_classes = spec['n_stat'](clf)['n_clusters']

        ax = axes[i]

        ax.set_axisbelow(True)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

        colors = cmap.colors[:n_classes]

        for i in range(n_classes):
            if not np.any(Y == i):
                continue

            ax.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=colors[i])

        ax.set_title(label + ': n=' + str(n_classes), fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

