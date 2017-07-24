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
    from scipy.signal import resample
    # Downsample
    ret = resample(x, num, axis=axis, **kwargs)
    # Upsample
    ret = resample(ret, x.shape[axis], axis=axis, **kwargs)

    return ret
