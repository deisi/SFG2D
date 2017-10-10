from numpy import median
from scipy.signal import resample

def double_resample(x, num, axis=0, **kwargs):
    """Use FFT bases resample twiche to FFT filter data and keep axis size constant.

    Parameter:
    x: array
    num: int
        donwsampling number.
    axis: int
        axis of x to apply the filter to. default is 0
    **kwargs get passed to both `scipy.signal.resample` method calls

    """
    # Downsample
    ret = resample(x, num, axis=axis, **kwargs)
    # Upsample
    ret = resample(ret, x.shape[axis], axis=axis, **kwargs)

    return ret

def replace_pixel(record, pixel, region=5):
    """Replace a given pixel in the raw data of record by a median of its surrounding.

    Usefull to cope with broken pixels.
    """

    record._rawData[:, :, :, pixel] = median(
        record._rawData[:, :, :, pixel-region: pixel+region], -1
    )
    return record
