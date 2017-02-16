"""Module for Widgets."""

import warnings, os
from glob import glob

import numpy as np
from scipy.signal import medfilt2d, medfilt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec

from .core import SfgRecord
from .io.veronica import pixel_to_nm
from .utils.static import nm_to_ir_wavenumbers
from .utils.consts import X_PIXEL_INDEX, Y_PIXEL_INDEX, SPEC_INDEX, FRAME_AXIS_INDEX, PP_INDEX, PIXEL

debug = 0

class WidgetBase():
    """A Base class for my widgets.

    Uses SfgRecord object as data container.
    Consists out of several ipywidgets.
    Plots are rendered using matplotlib.

    Define any ipwidget you need within this class, within the
    *WidgetBase._init_widget* function. Default or context
    dependet options of the widgets can be set during the
    *WidgetBase._configure_widgets* function.
    The observers of the widgets are set within the *_init_observer*
    function, or if it is an figure updating widget within the
    *_init_figure_observers* function.
    If an observer is defined, also define an unobserver in the
    *_unobserver* function.
    """
    def __init__(self, data=SfgRecord(), fig=None, ax=None,
                 central_wl=None, vis_wl=None, figsize=None):
        # SfgRecord obj holding the data.
        self.data = data
        # 4 dim numpy array representing the baseline
        self.data_base = np.zeros_like(self.data.data, dtype="int64")

        # Internal objects
        self._fig = fig
        self._central_wl = central_wl
        self._vis_wl = vis_wl
        self._figsize = figsize
        self._figure_widgets = [] # List of widgets that update the figure
        self.children = [] # List of widgets to display

        # Setup all widgets
        self._init_widget()

    def __call__(self):
        """Use call to actually Render the widgets on the notebook."""
        from IPython.display import display
        self._configure_widgets()
        self._init_observer()
        self._update_figure()
        display(self.children)
        self.fig

    def _init_widget(self):
        """Init all widgets.

        Add widgets within this function. If possible you can give default
        properties to the widgets already within this function. Also use
        this function to combine many widgets into e.g. boxes

        """
        # A widget property starts with w_.
        # self.w_children is going to be displayed. All widgets that should be
        # visible must register with that.
        # Its not too bad if you have more widgets then you need.
        # only those in self.children are rendered. The rest is ignored.
        import ipywidgets as wi

        ### Widgets that do something
        self.wTextFolder = wi.Text(
            layout = wi.Layout(width='80%'),
        )
        self.wSelectFile = wi.Select(
            descripion='Files',
            layout = wi.Layout(width='100%'),
        )
        self.wSelectBaseFile = wi.Select(
            description='Base',
            layout=self.wSelectFile.layout,
        )
        self.wToggleSubBaseline = wi.ToggleButton(
            description='Sub Baseline',
            layout=wi.Layout(width='70%'),
        )
        self.wCheckShowBaseline = wi.Checkbox(
            description='Show Baseline',
            value=True
        )
        self.wIntSliderSmooth = wi.IntSlider(
            continuous_update=False, description="smooth",
            min=1, max=19, step=2,
        )
        self.wCheckAutoscale = wi.Checkbox(
            description="Autoscale",
            value=True,
        )
        self.wIntRangeSliderPixelY = wi.IntRangeSlider(
            continuous_update=False, description="y_pixels"
        )
        self.wIntRangeSliderPixelX = wi.IntRangeSlider(
            continuous_update=False, description="x_pixels",
            max = PIXEL, value=(int(PIXEL*0.25), int(PIXEL*0.75)),
        )
        self.wTextCentralWl = wi.FloatText(
            description='central wl', value=self.central_wl,
            layout=wi.Layout(width='10%', margin='2px 50px 2px 0px')
        )
        self.wDropdownCalib = wi.Dropdown(
            description='x-axis', options=['pixel', 'nm', 'wavenumber'],
            layout=wi.Layout(width='60px', margin = "0px 130px 0px 0px"),
        )
        self.wTextVisWl = wi.FloatText(
            description='vis wl', value=self.vis_wl,
            layout=self.wTextCentralWl.layout
        )
        self.wSliderPPDelay = wi.SelectionSlider(
            continuous_update=False, description="pp_delay",
        )
        self.wSliderFrame = wi.IntSlider(
            continuous_update=False, description="frame"
        )
        self.wCheckFrameMedian = wi.Checkbox(
            description='median',
        )
        self.wSliderBaselinePPDelay = wi.IntSlider(
            description='pp_delay index', continuous_update=False)
        self.wSliderBaselineFrame = wi.IntSlider(
            description='frame', continuous_update=False)
        self.wCheckBaselineFrameMedian = wi.Checkbox(
            description='median', value=False)
        self.wRangeSliderBaselineSpec = wi.IntRangeSlider(
            description='spectrum', continuous_update=False)
        self.wDropSumAxis = wi.Dropdown(
            description='sum x-axis',
            options=('pp_delays', 'frames'),
            layout=wi.Layout(width='60px'),
        )
        self.wTextBaselineOffset = wi.FloatText(
            description='Offset', value=0,
            layout=wi.Layout(widht = "10px"),
        )

        ### Aligning boxers ###
        folder_box = wi.HBox([wi.Label("Folder",
                              layout=wi.Layout(margin='0px 123px 0px 0px'))
                              , self.wTextFolder])
        self.wVBoxData = wi.VBox(
            [
                wi.Label("Data:"),
                folder_box,
                wi.HBox([
                    wi.VBox([self.wToggleSubBaseline, self.wCheckShowBaseline, self.wCheckAutoscale,]),
                    self.wSelectFile, self.wSelectBaseFile,
                ]),
            ],
            layout=wi.Layout(margin = '2px 0px 16px 0px')
        )
        #self.wVBoxData.border = '1px black solid'

        self.wVBoxSignal = wi.VBox(
            [
                wi.Label("Spectrum:"),
                wi.HBox([self.wSliderPPDelay, self.wIntSliderSmooth, self.wCheckFrameMedian]),
                wi.HBox([self.wSliderFrame, self.wIntRangeSliderPixelY, self.wIntRangeSliderPixelX]),
            ],
            layout=self.wVBoxData.layout
        )
        #self.wVBoxSignal.border = "1px black solid"

        self.wVBoxBaseline = wi.VBox([
            wi.Label("Baseline:"),
            wi.HBox([
                self.wSliderBaselinePPDelay, self.wSliderBaselineFrame,
                self.wRangeSliderBaselineSpec, self.wCheckBaselineFrameMedian, ]),
                self.wTextBaselineOffset
            ],
            layout=self.wVBoxData.layout
        )

        # Widgets that update the figure on value change
        self._figure_widgets = [
            self.wSelectFile,
            self.wDropdownCalib,
            self.wSliderPPDelay,
            self.wSliderFrame,
            self.wCheckFrameMedian,
            self.wIntRangeSliderPixelY,
            self.wIntRangeSliderPixelX,
            self.wTextVisWl,
            self.wTextCentralWl,
            self.wCheckAutoscale,
            self.wIntSliderSmooth,
            self.wToggleSubBaseline,
            self.wCheckShowBaseline,
            self.wSliderBaselinePPDelay,
            self.wSliderBaselineFrame,
            self.wRangeSliderBaselineSpec,
            self.wCheckBaselineFrameMedian,
            self.wDropSumAxis,
            self.wTextBaselineOffset,
        ]

    def _configure_widgets(self):
        """Set all widget options. And default values.

        It must be save to recall this method at any time."""

        self.wSliderPPDelay.options = self.data.pp_delays.tolist()
        if self.wSliderPPDelay.value not in self.wSliderPPDelay.options:
            self.wSliderPPDelay.value = self.wSliderPPDelay.options[0]
        if self.data.pp_delays.shape == (1,):
            self.wSliderPPDelay.disabled = True
        else:
            self.wSliderPPDelay.disabled = False

        self.wSliderFrame.max = self.data.data.shape[FRAME_AXIS_INDEX] - 1
        if self.data.data.shape[FRAME_AXIS_INDEX] == 1:
            self.wSliderFrame.disabled = True
        else:
            self.wSliderFrame.disabled = False
        self.wIntRangeSliderPixelY.max = self.data.data.shape[Y_PIXEL_INDEX]
        if np.any(np.array(self.wIntRangeSliderPixelY.value) > self.wIntRangeSliderPixelY.max):
            self.wIntRangeSliderPixelY.value = self.wIntRangeSliderPixelY.min, self.wIntRangeSliderPixelY.max
        if self.data.data.shape[Y_PIXEL_INDEX] == 1:
            self.wIntRangeSliderPixelY.disabled = True
        else:
            self.wIntRangeSliderPixelY.disabled = False
        self.wIntRangeSliderPixelX.max = self.data.data.shape[X_PIXEL_INDEX]
        if np.any(np.array(self.wIntRangeSliderPixelX.value) > self.wIntRangeSliderPixelX.max):
            self.wIntRangeSliderPixelX.value = self.wIntRangeSliderPixelX.min, self.wIntRangeSliderPixelX.max

        if isinstance(self.central_wl, type(None)):
            self.wTextCentralWl.value = 0
        else:
            self.wTextCentralWl.value = self.central_wl
        self.wTextVisWl.value = self.vis_wl
        self.wSliderBaselinePPDelay.max = self.data_base.shape[PP_INDEX] - 1
        if self.wSliderBaselinePPDelay.max is 0:
            self.wSliderBaselinePPDelay.disabled = True
        else:
            self.wSliderBaselinePPDelay.disabled = False
        if self.wSliderBaselinePPDelay.value > self.wSliderBaselinePPDelay.max:
            self.wSliderBaselinePPDelay.value = 0
        self.wSliderBaselineFrame.max = self.data_base.shape[FRAME_AXIS_INDEX] - 1
        if self.wSliderBaselineFrame.max is 0:
            self.wSliderBaselineFrame.disabled = True
        else:
            self.wSliderBaselineFrame.disabled = False
        if self.wSliderBaselineFrame.value >  self.wSliderBaselineFrame.max:
            self.wSliderBaselineFrame.value = 0
        self.wRangeSliderBaselineSpec.max = self.data_base.shape[Y_PIXEL_INDEX]
        self.wRangeSliderBaselineSpec.min = 0
        if self.wRangeSliderBaselineSpec.max is 0:
            self.wRangeSliderBaselineSpec.disabled = True
        else:
            self.wRangeSliderBaselineSpec.disabled = False
        if self.wRangeSliderBaselineSpec.value[1] > self.wRangeSliderBaselineSpec.max:
            self.wRangeSliderBaselineSpec.value[1] = self.wRangeSliderBaselineSpec.max
            self.wRangeSliderBaselineSpec.value[0] = 0

        self._toggle_central_wl()
        self._toggle_vis_wl()
        self._toggle_frame_slider()
        self._toggle_sum_over()

    def _init_figure(self):
        """Init figure.

        This will be most likely  be overwritten by the children."""
        if not self.fig:
            self._fig = plt.gcf()

        if len(self.axes) is 0:
            self._fig.add_subplot(111)

    def _update_figure(self):
        """Update figure with changes"""
        # MUST OVERWRITE THIS
        pass

    def _update_figure_callback(self, new):
        """A callback version of _update_figer for usage in observers."""
        self._update_figure()

    def _init_figure_observers(self):
        """All observers that call the *update_figure_callback* """

        # Because during widget runtime it can be necessary to stop
        # and restart the automatic figure updating to prevent flickering
        # and to speed up the gui.
        for widget in self._figure_widgets:
            widget.observe(self._update_figure_callback, "value")

    def _init_observer(self):
        """Set all observer of all subwidgets."""
        # This registers the callback functions to the gui elements.
        # After a call of _init_observer, the gui elements start to
        # actually do something, namely what ever is defined within the
        # callback function of the observer.
        self.wTextFolder.on_submit(self._on_folder_submit)
        self.wSelectFile.observe(self._update_data, 'value')
        self.wDropdownCalib.observe(self._toggle_vis_wl, 'value')
        self.wDropdownCalib.observe(self._toggle_central_wl, 'value')
        self.wCheckFrameMedian.observe(self._toggle_frame_slider, 'value')
        self.wSelectBaseFile.observe(self._on_base_changed, 'value')
        self.wCheckBaselineFrameMedian.observe(self._toggle_frame_base_slider, 'value')
        self._init_figure_observers()

    def _unobserve_figure(self):
        """Unobserver figure observers."""
        for widget in self._figure_widgets:
            widget.unobserve(self._update_figure_callback, 'value')

    def _on_folder_submit(self, new):
        """Called when folder is changed."""
        fnames = _filter_fnames(self.wTextFolder.value)

        if debug:
            print("_on_folder_submit_called")
        if debug > 1:
            print("fnames:", fnames)

        # The with is a workaround. I need it in the test functions,
        # not the gui. Anyways, it doesn't quite work.
        with self.wSelectFile.hold_trait_notifications():
            self.wSelectFile.options = fnames
        with self.wSelectBaseFile.hold_trait_notifications():
            self.wSelectBaseFile.options = self.wSelectFile.options

    def _toggle_vis_wl(self, new=None):
        """Toggle the vis wl text box according to calibration axis.

        The new keyword exists, so it can also server as a callback function."""
        if self.wDropdownCalib.value == 'wavenumber':
            self.wTextVisWl.disabled = False
        else:
            self.wTextVisWl.disabled = True

    def _toggle_central_wl(self, new=None):
        """Toggle the central wl text box according to calibration axis.

        The new keyword exists, so it can also server as a callback function."""
        if self.wDropdownCalib.value == 'pixel' or self.data._type == 'spe':
            self.wTextCentralWl.disabled = True
        else:
            self.wTextCentralWl.disabled = False

    def _toggle_frame_slider(self, new=None):
        """Toggle frame slider according to number of frames.

        The new keyword exists, so it can also server as a callback function."""
        if self.wCheckFrameMedian.value:
            self.wSliderFrame.disabled = True
        else:
            self.wSliderFrame.disabled = False

    def _toggle_frame_base_slider(self, new=None):
        """Toggle frame slider of the base/background according to frames.

        The new keyword exists, so it can also server as a callback function."""
        if self.wCheckBaselineFrameMedian.value:
            self.wSliderBaselineFrame.disabled = True
        else:
            self.wSliderBaselineFrame.disabled = False

    def _toggle_sum_over(self, new=None):
        if self.data.frames is 1:
            self.wDropSumAxis.value = "pp_delays"
            self.wDropSumAxis.disabled = True
            return
        if self.data.pp_delays.shape[0] is 1:
            self.wDropSumAxis.value = "frames"
            self.wDropSumAxis.disabled = True
            return
        self.wDropSumAxis.disabled = False

    def _update_data(self, new):
        """Update the internal data objects.

        The internal data objects are updated according to
        *WidgetBase.wTextFolder.value* and *WidgetBase.wSelectFile.value*.
        The *WidgetBase._central_wl* property gets reseted, and child
        widget elements, like e.g. WidgetBase.wSliderPPDelay are checked for
        correctness and reseted."""
        fname = self.wTextFolder.value + "/" + self.wSelectFile.value
        self.data = SfgRecord(fname)
        self._central_wl = None
        # Deactivating the observers here prevents flickering
        # and unneeded calls of _update_figure. Thus we
        # call it manually after a recall of _init_observer
        self._unobserve_figure()
        self._configure_widgets()
        self._init_figure_observers()
        self._update_figure()

    def _on_base_changed(self, new):
        """Change the data file of the baseline.

        Resets all elements that need to be reseted on a baseline change."""
        fname = self.wTextFolder.value + "/" + self.wSelectBaseFile.value
        # If we have already loaded the data to ram for the baseline,
        # we just copy it from there. Thats a lot faster then reading
        # it from the HDD again.
        if self.wSelectBaseFile.value is self.wSelectFile.value:
            if debug:
                print("Copied new baseline data from ram.")
            self.data_base = self.data.data.copy()
        else:
            self.data_base = SfgRecord(fname).data
        # Deactivating the observers here prevents flickering
        # and unneeded calls of _update_figure. Thus we
        # call it manually after a recall of _init_observer
        self._unobserve_figure()
        self._configure_widgets()
        self._init_figure_observers()
        self._update_figure()

    @property
    def fig(self):
        return self._fig

    @property
    def axes(self):
        return self._fig.axes

    def _calibrate_x_axis(self, data, wavelength=None, wavenumber=None):
        """Apply calibrations to x axis."""

        # An importat feature of a calibration is, that it doesnt
        # change the shape.

        # Need pixel as a possible default value.
        pixel = np.arange(data.shape[X_PIXEL_INDEX])
        central_wl = self.wTextCentralWl.value
        vis_wl = self.wTextVisWl.value

        if self.wDropdownCalib.value == 'pixel':
            x = pixel
        if self.wDropdownCalib.value == 'nm':
            x = wavelength
            if isinstance(wavelength, type(None)):
                x = np.arange(len(pixel))
            if self.wTextCentralWl.value != 0:
                x = pixel_to_nm(pixel, central_wl)
        if self.wDropdownCalib.value == 'wavenumber':
            x = wavenumber
            if isinstance(wavenumber, type(None)):
                x = pixel[::-1]
            if central_wl == 0 and vis_wl != 0 and not isinstance(wavelength, type(None)):
                x = nm_to_ir_wavenumbers(wavelength, vis_wl)
            if central_wl != 0 and vis_wl != 0:
                nm = pixel_to_nm(pixel, central_wl)
                x = nm_to_ir_wavenumbers(nm, vis_wl)
        return x

    @property
    def x(self):
        """X data of the *Signal* plot. """

        # Shape changing transformations belong in here.

        wavelength = getattr(self.data, 'wavelength', None)
        wavenumber = getattr(self.data, 'wavenumber', None)
        ret = self._calibrate_x_axis(
            self.data.data,
            wavelength,
            wavenumber
        )
        return ret

    def _prepare_y_data(self, data):
        """Non shape changing transformations."""
        #TODO Refactor with better name
        #TODO Test if copy is still needed.
        y = data.copy()
        y = y.astype('float64')
        if self.wToggleSubBaseline.value:
            if len(self.y_base.shape) == 1:
                y -= np.ones_like(y) * self.y_base
            elif self.y_base.shape[1] == 1:
                y -= np.ones_like(y) * self.y_base[:, 0]
            else:
                # Check if this is possible
                if self.data_base.shape[SPEC_INDEX] is not self.data.data.shape[SPEC_INDEX]:
                    message = 'Cant subtract baseline spectra wise due to' \
                              'unmatched dimensions. Data shape is %s, but' \
                              'baseline shape is %s' %\
                               (self.data.data.shape, self.data_base.shape)
                    warnings.warn(message)
                    self.wToggleSubBaseline.value = False
                    return y
                base = self.data_base[self.wSliderBaselinePPDelay.value, :]
                if self.wCheckBaselineFrameMedian.value:
                    base = np.median(base, 0)
                else:
                    base = base[self.wSliderBaselineFrame.value]
                y -= base[None, None, :] + self.wTextBaselineOffset.value
        return y

    @property
    def y(self):
        """Y data of the *Signal* plot."""

        #TODO the ppdelay slider needs to be a custom slider.
        pp_delays = getattr(self.data, 'pp_delays')
        y_slice = _slider_range_to_slice(self.wIntRangeSliderPixelY.value,
                                         self.data.data.shape[Y_PIXEL_INDEX])
        pp_delay_index = np.where(
            self.wSliderPPDelay.value == pp_delays)[0][0]

        # TODO add the possibility to select frame regions.
        ret =  self._prepare_y_data(self.data.data)[pp_delay_index, :, :, :]
        if self.wCheckFrameMedian.value:
            ret = np.median(ret, FRAME_AXIS_INDEX)
        else:
            ret = ret[
                self.wSliderFrame.value,
                :,
                :
            ]
        ret = ret[y_slice, :]
        # Must be done here, because it works only on 2d data.
        # TODO I could use the not 2d version
        if self.wIntSliderSmooth.value != 1:
            ret = medfilt2d(ret, (1, self.wIntSliderSmooth.value))
        return ret.T

    @property
    def x_base(self):
        """x data of the baseline in the *Signal* plot"""
        wavelength = getattr(self.data, 'wavelength', None)
        wavenumber = getattr(self.data, 'wavenumber', None)
        x = self._calibrate_x_axis(
            self.data_base,
            wavelength,
            wavenumber
        )
        return x

    @property
    def y_base(self):
        """y data of the baseline in the *Signal* plot."""

        # TODO this is a hack, because I cant select frame regions yet
        frame_slice = self.wSliderBaselineFrame.value
        spec_slice = _slider_range_to_slice(
            self.wRangeSliderBaselineSpec.value,
            self.wRangeSliderBaselineSpec.max,
        )
        if self.wCheckBaselineFrameMedian.value:
            data = self.data_base[
                self.wSliderBaselinePPDelay.value,
                :,
                spec_slice,
                :]
            y = np.median(data, 0) + self.wTextBaselineOffset.value
        else:
            y = self.data_base[
                    self.wSliderBaselinePPDelay.value,
                    frame_slice,
                    spec_slice,
                    :] + self.wTextBaselineOffset.value
        return y.T

    @property
    def central_wl(self):
        """Central wl used for x axis calibration of the *Spectrum* axis."""
        if self._central_wl == None:
            if self.data.metadata.get('central_wl') != 0:
                self._central_wl = self.data.metadata.get('central_wl')
        return self._central_wl

    @property
    def vis_wl(self):
        """The wavelength of the visible.

        The visible wavelength is used as upconversion number during the
        calculation of the wavenumber values of the x axis of the *Signal* plot."""
        if self._vis_wl == None:
            return 0
        return self._vis_wl


class SpecAndBase(WidgetBase):
    """A widget that allows for selection of a spectrum and baseline plus visualization."""

    def __init__(self, figsize=(8 , 6), **kwargs):
        super().__init__(figsize=figsize, **kwargs)

    def __call__(self):
        super().__call__()

    def _init_widget(self):
        """Init the widgets that are to be shown."""
        import ipywidgets as wi
        super()._init_widget()
        # This allows the data to be used for normalization from start on
        self.data.data += 1
        self.wIntRangeSliderPixelX.layout.visibility = 'hidden'
        self.children = wi.VBox([
                self.wVBoxData,
                self.wVBoxSignal,
                self.wVBoxBaseline,
                wi.HBox([self.wDropdownCalib, self.wTextCentralWl, self.wTextVisWl]),
        ])

    def _init_figure(self):
        """Init the fiures and axes"""
        if not self._fig:
            self._fig, ax = plt.subplots(1, 1, figsize=self._figsize)
        # This allows for redrawing the axis on an already existing figure.
        elif self._fig and len(self.axes) is not 1:
            self._fig.set_size_inches(self._figsize, forward=True)
            self._ax = self._fig.add_subplot(111)

    def _update_figure(self):
        """Is called on all gui element changes.

        This function renders the plot. When ever you want to make changes
        visible in the figure you need to run this."""
        self._init_figure()
        ax = self.axes[0]
        ax.clear()
        ax.plot(self.x, self.y, label='Spectrum')
        if self.wCheckShowBaseline.value:
            ax.plot(self.x_base, self.y_base, label='Baseline')
        ax.legend(framealpha=0.5)
        if self.wCheckAutoscale.value:
            self._ax_xlim = ax.get_xlim()
            self._ax_ylim = ax.get_ylim()
        if not isinstance(self._ax_xlim, type(None)):
            ax.set_xlim(*self._ax_xlim)
        if not isinstance(self._ax_ylim, type(None)):
            ax.set_ylim(*self._ax_ylim)


class SpecAndSummed(WidgetBase):
    def __init__(self, central_wl=None, vis_wl=810, figsize=(10, 4), **kwargs):
        """Plotting gui based on the SfgRecord class as a data backend.

        Parameters
        ----------
        data : Optional, SfgRecord obj.
            Default dataset to start with. If None is given, an empty one is created.
        fig: Optional, matplotlib figure
            The figure to draw on.
            Defaults to create a new one.
        ax: Optional, matplotlib axes. The axes to draw on.
            Defaults to create a new one.
        central_wl: float
            Central wavelength of the camera to start with.
            If none is given, it tryes to find out from by investigating the
            metadata.
        vis_wl: float
            Wavelength of the visible to begin with.

        Example:
        -------
        test = SpecAndSummed()
        test()
        # Type the Folder you want to investigate in the folder Text box and press RETURN.
        # A list of selectable files will appear on the right side.
        """
        super().__init__(central_wl=central_wl, vis_wl=vis_wl, figsize=figsize,
                         **kwargs)
        self._ax_xlim = None
        self._ax_ylim = None

    def _init_figure(self):
        """Init the two axis figure."""
        if not self._fig:
            self._fig, ax = plt.subplots(1, 2, figsize=self._figsize)
        # This allows for redrawing the axis on an already existing figure.
        elif self._fig and len(self.axes) is not 2:
            self._fig.set_size_inches(self._figsize, forward=True)
            [self._fig.add_subplot(121), self._fig.add_subplot(122)]

    def _init_widget(self):
        """Init all widgets that are to be drawn."""
        import ipywidgets as wi
        super()._init_widget()
        # self.children is the widget we are rendering up on call.

        self.children = wi.VBox([
                self.wVBoxData,
                self.wVBoxSignal,
                self.wVBoxBaseline,
                wi.HBox([self.wDropdownCalib, self.wTextCentralWl, self.wTextVisWl]),
                self.wDropSumAxis,
        ])

    def _update_figure(self):
        """Update the figure of the gui.

        Gets called on all button changes."""
        self._init_figure()
        fontsize = 8
        ax = self.axes[0]
        ax.clear()
        ax.plot(self.x, self.y, label='Spectrum')
        if self.wCheckShowBaseline.value:
            ax.plot(self.x_base, self.y_base, label='Baseline')
        # Buffer ax_{x,y}_lim so we can reuse it later
        if self.wCheckAutoscale.value:
            self._ax_xlim = ax.get_xlim()
            self._ax_ylim = ax.get_ylim()
        if not isinstance(self._ax_xlim, type(None)):
            ax.set_xlim(*self._ax_xlim)
        if not isinstance(self._ax_ylim, type(None)):
            ax.set_ylim(*self._ax_ylim)
        ax.legend(framealpha=0.5)
        ax.set_xlabel(self.wDropdownCalib.value)
        ax.set_title('Spectrum')
        ax.vlines(self.x_vlines, *ax.get_ylim(),
                  linestyle="dashed")
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.set_title("Signal")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

        ax = self.axes[1]
        ax.clear()
        y_sum = self.y_sum
        lines0_1 = ax.plot(self.x_sum, y_sum)
        points0_1 = ax.plot(self.x_sum, y_sum, 'o')
        for i in range(len(lines0_1)):
            points = points0_1[i]
            color = lines0_1[i].get_color()
            points.set_color(color)
        if "pp_delays" in self.wDropSumAxis.value:
            ax.set_xlabel('pp delay / fs')
        else:
            ax.set_xlabel("# frame")
        ax.set_ylabel('Sum')
        ax.set_title('Summed')
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3g'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax.yaxis.tick_right()

        self._fig.canvas.draw()
        for ax in self.axes:
            ax.figure.canvas.draw()

    @property
    def y_sum(self):
        """y data of the summed plot."""
        y_slice = _slider_range_to_slice(self.wIntRangeSliderPixelY.value,
                                         self.data.data.shape[Y_PIXEL_INDEX])
        x_slice = _slider_range_to_slice(self.wIntRangeSliderPixelX.value,
                                         self.data.data.shape[X_PIXEL_INDEX])

        pp_delays = getattr(self.data, 'pp_delays')
        pp_delay_index = np.where(
            self.wSliderPPDelay.value == pp_delays)[0][0]

        y = self._prepare_y_data(self.data.data)
        y = y[:, :, y_slice, x_slice]


        if 'pp_delays' in self.wDropSumAxis.value:
            if self.wCheckFrameMedian.value:
                y = np.median(y, FRAME_AXIS_INDEX)
            else:
                y = y[:, self.wSliderFrame.value, :, :]
        if 'frames' in self.wDropSumAxis.value:
            y = y[pp_delay_index, :, :, :]

        if self.wIntSliderSmooth.value is not 1:
            y = medfilt(y, (1, 1, self.wIntSliderSmooth.value))

        return y.sum(X_PIXEL_INDEX)

    @property
    def sum_y(self):
        warnings.warn("sum_y is deprecated plz use y_sum")
        return self.y_sum

    @property
    def x_sum(self):
        """x data of the summed plot."""
        #print(self.wDropSumAxis.value)
        if 'pp_delays' in self.wDropSumAxis.value:
            return self.data.pp_delays
        elif 'frames' in self.wDropSumAxis.value:
            return np.arange(self.data.frames)
        raise NotImplementedError('got %s for wDropSumAxis' % self.wDropSumAxis.value)

    @property
    def sum_x(self):
        warnings.warn("sum_x is deprecated. Use x_sum")
        return x_sum

    @property
    def x_vlines(self):
        ret = [self.x[self.wIntRangeSliderPixelX.value[0]],
               self.x[self.wIntRangeSliderPixelX.value[1] - 1]]
        return ret

    def _on_xlim_changed(self, new=None):
        """Callback for the *Signal* axis."""
        # Called when the xlim of the *Signal* plot is changed
        self._ax_xlim = self.axes[0].get_xlim()

    def _on_ylim_changed(self, new=None):
        # Called when the ylim of the *Signal* plot is cahnged
        self._ax_ylim = self.axes[0].get_ylim()

# TODO Find Better name
class Normalized(SpecAndSummed):
    """Widget to visualize data after normalization."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_widget(self):
        import ipywidgets as wi
        super()._init_widget()
        self.wCheckShowBaseline.value = False
        self.children = wi.VBox([
            #self.w_reload,
            wi.HBox([self.wSliderPPDelay, self.wIntSliderSmooth, self.wCheckAutoscale]),
            wi.HBox([self.wSliderFrame, self.wIntRangeSliderPixelY, self.wIntRangeSliderPixelX, self.wCheckFrameMedian,
                  #self.wIntRangeSliderPixelX
            ]),
            wi.HBox([self.wDropdownCalib, self.wDropSumAxis]),
            self.wTextCentralWl, self.wTextVisWl,
        ])

class ImgView(WidgetBase):
    """A Class to view full spe images."""
    def __init__(self, *args, figsize=(8,6), **kwargs):
        super().__init__(*args, figsize=figsize, **kwargs)

    def _init_figure(self):
        if not self._fig:
            self._fig = plt.figure(self._figsize)
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3],
                                   height_ratios=[3,1])
            ax = self._fig.add_subplot(gs[0, 1])
            self._fig.add_subplot(gs[0, 0], sharey=ax)
            self._fig.add_subplot(gs[1, 1], sharex=ax)
        elif self._fig and len(self.axes) is not 3:
            self._fig.set_size_inches(self._figsize, forward=True)
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3],
                                   height_ratios=[3,1])
            ax = self._fig.add_subplot(gs[0, 1])
            self._fig.add_subplot(gs[0, 0], sharey=ax)
            self._fig.add_subplot(gs[1, 1], sharex=ax)


    def _update_figure(self):
        self._init_figure()
        view_data = self.data.data[self.wSliderPPDelay.value, self.wSliderFrame.value]
        ax = self.axes[0]
        plt.sca(ax)
        ax.clear()
        img = ax.imshow(view_data, interpolation=self.w_interpolate.value,
                  origin="lower", aspect="auto")
        plt.colorbar(img)

        axl = self.axes[1]
        axl.clear()
        y_slice = _slider_range_to_slice(self.wIntRangeSliderPixelY.value,
                                         self.data.data.shape[Y_PIXEL_INDEX])
        view_data2 = self.data.data[self.wSliderPPDelay.value, self.wSliderFrame.value, y_slice].sum(Y_PIXEL_INDEX)
        axl.plot(view_data2)

    def _init_widget(self):
        import ipywidgets as wi
        super()._init_widget()
        self.wIntSliderSmooth.visible = False
        self.wIntSliderSmooth.disabled  = True
        self.wSliderPPDelay.visible = False
        self.wSliderPPDelay.disabled = True
        self.w_interpolate = wi.Dropdown(
            description="Interpolation",
            options=('none', 'nearest', 'bilinear', 'bicubic',
                     'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
                     'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
                     'lanczos'),
            value = "nearest",
        )
        self.children = wi.VBox([
            self.wVBoxSignal,
            wi.HBox([self.wDropdownCalib, self.wTextCentralWl, self.wTextVisWl]),
            self.w_interpolate,
        ])

    def _init_observer(self):
        super()._init_observer()
        self.w_interpolate.observe(self._update_figure_callback, "value")


class Dashboard():
    def __init__(self, *args, **kwargs):
        self.widgets = args
        self.fig = None
        self.ax = None

    def __call__(self):
        pass

class PumpProbe(Dashboard):
    """Tabed dashboard.

    The first page shows two axis.
    On the first axes one sees the raw signal. And possibly
    a baseline. Each y-pixel of the ccd camera gets projected into a single.
    spectra line on this first axes. With the *Show Baseline* Button one can
    toggle the visibility of the Baseline. Autoscale prevents the axes from
    re-scaling up on data change. Numorus sliders allow for inspection of the
    data.
    The second axes shows the Sum of each spectrum vs pump-probe time delay.
    This is only use full if you do pump-probe experiment. Otherwise this axis
    will only show to you the a single point with the value of the sum(area) of
    the spectrum from axes one.


    The second page shows A single Spectrum and possibly a baseline.

    The third page shows, after usage of the normalize button the quotient
    of the first and the second page spectrum. This allows for IR Normalization."""

    def __init__(self, *args, **kwargs):
        import ipywidgets as wi
        super().__init__(*args, **kwargs)
        children = []
        self.wi_fig = plt.figure()
        for widget in args:
            widget._configure_widgets()
            children.append(widget.children)
            widget._fig = self.wi_fig

        self.w_tabs = wi.Tab(children=children)
        self.children = self.w_tabs
        self.w_normalize = wi.Button(description='Normalize')
        self.children = wi.VBox([self.w_tabs, self.w_normalize])

    def __call__(self):
        from IPython.display import display
        for widget in self.widgets:
            widget._init_observer()
        self._init_observer()
        display(self.children)

    def _init_observer(self):
        if debug:
            print("Dasboards._init_observer called")
        self.w_tabs.observe(self._on_tab_changed, 'selected_index')
        self.w_normalize.on_click(self._on_normalize)

        # observers to w? must be re initiated on each data change.
        w0, w1, *_ = self.widgets
        # Need to make shure, that Normalize button is only clickable,
        # When the shape of the data allows for normalization
        w0.wIntRangeSliderPixelY.observe(self._is_normalizable_callback, "value")
        w1.wIntRangeSliderPixelY.observe(self._is_normalizable_callback, "value")

    def _on_tab_changed(self, new):
        if debug:
            print("Dashboard._on_tab_changed called")
        axes = self.wi_fig.axes
        for ax in axes:
             self.wi_fig.delaxes(ax)
        page = self.w_tabs.selected_index
        widget = self.widgets[page]
        if page == 3:
            self.widgets[3].data = self.widgets[0].data
            self.widgets[3]._configure_widgets()
        widget._update_figure()

    def _on_normalize(self, new):
        if debug:
            print("Normalize._on_normalize called.")
        if not self._is_normalizable:
            return

        w0, w1, w2, *_ = self.widgets
        spec = w0._prepare_y_data(w0.data.data)
        #TODO add frame_med filter here.
        ir = np.ones_like(spec) * w1.y.T

        w2.data = w0.data.copy()
        w2.data.data = spec/ir
        w2._unobserve_figure()
        if self.w_tabs.selected_index is 2:
            w2._update_figure()
        w2._configure_widgets()
        w2._init_figure_observers()

    @property
    def _is_normalizable(self):
        w0, w1, *_  = self.widgets
        if w0.y.shape[0] != w1.y.shape[0]:
            self.w_normalize.disable = True
            return False
        if w1.y.shape[1] is 1:
            self.w_normalize.disabled = False
            return True
        if w0.y.shape[1] != w1.y.shape[1]:
            self.w_normalize.disabled = True
            return False
        self.w_normalize.disabled = False
        return True

    def _is_normalizable_callback(self, new):
        self._is_normalizable


##### Helper function
def _filter_fnames(folder_path):
    """Return list of known files in a folder."""

    fnames = np.sort(glob(os.path.normcase(folder_path + '/*' )))
    # Only .dat and .spe in known
    mask = [a or b for a, b in zip([".dat" in s for s in fnames], [".spe" in s for s in fnames])]
    fnames = fnames[np.where(mask)]
    # Remove AVG
    fnames = fnames[np.where(["AVG" not in s for s in fnames])]
    fnames = [os.path.split(elm)[1] for elm in fnames]
    return fnames


def _slider_range_to_slice(range_value_tuple, max):
    """Transform a tuple into a slice accounting for overlapping"""
    # This can happen if one links
    # sliders but can not guarantee that they
    # refer to the same data.
    if range_value_tuple == (0, 0):
        return slice(None, None, None)
    if any(range_value_tuple) > max:
            if range_value_tuple[0] > max:
                range_value_tuple[0] = max
            range_value_tuple[1] = range_value_tuple[0]

    if range_value_tuple[0] == range_value_tuple[1]:
        if range_value_tuple[0] < max:
            range_value_tuple = range_value_tuple[0], range_value_tuple[0] + 1
        else:
            range_value_tuple = range_value_tuple[1] - 1, range_value_tuple[1]

    return slice(*range_value_tuple)

#### End of helper functions
