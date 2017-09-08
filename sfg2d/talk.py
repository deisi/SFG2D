
"""Import this for Poster Settings"""
from matplotlib import rcParams
def configure():
    """
    Configure matplotlib, so it works well for posters.
    """
    rcParams['savefig.dpi'] = 150
    rcParams['figure.figsize'] = [8.0, 6.0]
    rcParams['font.size'] = 18
    rcParams['savefig.transparent'] = True
    rcParams['lines.linewidth'] = 4
    rcParams['lines.markersize'] = 8
    rcParams['legend.framealpha'] = 0
    return

if __name__ != "__main__":
    configure()
