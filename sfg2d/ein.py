from matplotlib import rcParams
def configure():
    """
    Configure matplotlib, so it works well within a EIN mode buffer.
    """

    rcParams['savefig.dpi'] = 80
    return

if __name__ != "__main__":
    configure()
