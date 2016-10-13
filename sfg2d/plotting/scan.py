import matplotlib.pyplot as plt


def sums(scan, fig=None, ax=None, norm=False, attribute_dict={}):
    """ Plot the sums of the scans"""
    if not fig:
        fig = plt.gcf()

    if not ax:
        ax = plt.gca()

    for name in scan.med.columns:
        data = scan.df[name].sum(0)
        if norm:
            data = data/data.max()
        ax.plot(data.values, label=name, **attribute_dict)
    plt.legend()
    plt.xlabel('Run')
    plt.ylabel('Area of spectrum')
    if norm:
        plt.ylabel('Normalized area of spectrum')
    plt.title('Area of spectrum over time')
    return fig, ax
