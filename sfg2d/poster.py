"""Import this for Poster Settings"""
from matplotlib import rcParams
def configure():
    """
    Configure matplotlib, so it works well for posters.
    """
    rcParams['savefig.dpi'] = 300
    rcParams['font.size'] = 32
    rcParams['savefig.transparent'] = True
    rcParams['lines.linewidth'] = 6
    rcParams['lines.markersize'] = 10
    rcParams['legend.framealpha'] = 0
    return

if __name__ != "__main__":
    configure()
