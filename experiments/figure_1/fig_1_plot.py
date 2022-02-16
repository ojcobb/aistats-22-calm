from scipy.stats.kde import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt


def ridgeline(data, W, overlap=0, fill=True, labels=None, n_points=150, alpha=0.1, adjustable=True):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(np.min(np.concatenate(data)),
                     np.max(np.concatenate(data)), n_points)
    ys = []
    quantile = np.quantile(data[0], 1-alpha)
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        if adjustable and i < W:
            quantile = np.quantile(d, 1-alpha)
        y = i*(1.0-overlap)
        ys.append(y)
        curve = pdf(xx)
        xs_below = xx[xx<quantile]
        xs_above = xx[xx>=quantile]
        if fill:
            plt.fill_between(xs_below, np.ones(len(xs_below))*y, 
                             curve[xx<quantile]+y, zorder=len(data)-i+1, color=fill)
            plt.fill_between(xs_above, np.ones(len(xs_above))*y, 
                             curve[xx>=quantile]+y, zorder=len(data)-i+1, color='#e60000')
        plt.plot(xx, curve+y, c='k', zorder=len(data)-i+1)
    if labels:
        plt.yticks(ys, labels)


def ridgeline_base(data, W, overlap=0, fill=True, labels=None, n_points=150, alpha=0.1, adjustable=True):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(np.min(np.concatenate(data)),
                     np.max(np.concatenate(data)), n_points)
    ys = []
    quantile = np.quantile(data[0], 1-alpha)
    for i in range(len(data)):
        y = i*(1.0-overlap)
        ys.append(y)
        if i>=(W-1):
            d = data[i-(W-1)]
            pdf = gaussian_kde(d)
            if adjustable and i < W:
                quantile = np.quantile(d, 1-alpha)
            curve = pdf(xx)
            xs_below = xx[xx<quantile]
            xs_above = xx[xx>=quantile]
            if fill:
                plt.fill_between(xs_below, np.ones(len(xs_below))*y, 
                                curve[xx<quantile]+y, zorder=len(data)-i+1, color=fill)
                plt.fill_between(xs_above, np.ones(len(xs_above))*y, 
                                curve[xx>=quantile]+y, zorder=len(data)-i+1, color='#e60000')
            plt.plot(xx, curve+y, c='k', zorder=len(data)-i+1)
        else:
            plt.plot(xx, np.zeros_like(xx), alpha=0.0)

    if labels:
        plt.yticks(ys, labels)


def main():

    N = 1000
    B = 100000
    W = 8
    extras = 4
    alpha = 0.1

    x_ref = np.random.randn(N)
    x_test = np.random.randn(B, 3*W-1)

    stats = []
    for i in range(W+extras):
        dists = x_ref.mean() - x_test[:,i:i+W].mean(-1)
        quantile = np.quantile(dists, 1-alpha)
        x_test = x_test[dists < quantile]
        stats.append(dists)

    plt.subplots(nrows=1, ncols=2, figsize=(7,3.5))

    plt.subplot(1,2,1)
    ridgeline_base(
        tuple(stats), W, labels=tuple((f't={i+1}' for i in range(W+extras))), 
        alpha=alpha, overlap=.1, fill='#00b33c')
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.xlabel(r'$S_t$', fontsize=16)
    plt.xlim((-1.12,1.12))
    plt.xticks([])
    plt.grid(zorder=0)
    plt.title('Fixed Threshold')


    plt.subplot(1,2,2)
    ridgeline(
        tuple(stats), W, labels=tuple((f't={i+1}' for i in range(W+extras))), 
        alpha=alpha, overlap=.1, fill='#00b33c')
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.xlabel(r'$S_t$', fontsize=16)
    plt.xlim((-1.12,1.12))
    plt.xticks([])
    plt.grid(zorder=0)
    plt.title('Ours')
    plt.tight_layout()
    plt.savefig('fig_1.pdf')

if __name__ == '__main__':
    main()