from matplotlib import rcParams
def configure():
    """
    Configure matplotlib, so it works well within a EIN mode buffer.
    """

    rcParams['savefig.dpi'] = 100
    return

if __name__ != "__main__":
    import seaborn as sns
    configure()
