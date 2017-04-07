"""Module for Widgets."""

import warnings, os
from glob import glob
from collections import Counter

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
        import ipywidgets as wi

        ### Any widget that we need at some point can be added here.

        # Widget to enter a folder path as string
        self.wTextFolder = wi.Text(
            layout = wi.Layout(width='80%'),
        )

        # Selection dialogue to select data from a list of files
        self.wSelectFile = wi.Select(
            descripion='Files',
            layout = wi.Layout(width='100%'),
        )

        # Selection dialogue to select baseline data from a list of files
        self.wSelectBaseFile = wi.Select(
            description='Base',
            layout=self.wSelectFile.layout,
        )

        # Toggle button to toggle the subtraction of the baseline data
        self.wToggleSubBaseline = wi.ToggleButton(
            description='Sub Baseline',
            layout=wi.Layout(width='70%'),
        )

        # Checkbox to toggle the visibility of raw spectra data
        self.wCheckShowSpectra = wi.Checkbox(
            description='Spectra',
            value=True
        )

        # Checkbox to toggle the visibility of the baseline data
        self.wCheckShowBaseline = wi.Checkbox(
            description='Baseline',
            value=True
        )

        # Checkbox to toggle the visibility of bleach data
        self.wCheckShowBleach = wi.Checkbox(
            description='Bleach',
            value=True
        )

        # Slider to select the width of the smoothing kernel
        self.wIntSliderSmooth = wi.IntSlider(
            continuous_update=False, description="smooth",
            min=1, max=19, step=2,
        )

        # Slider to select smoothing of baseline
        self.wIntSliderSmoothBase = wi.IntSlider(
            continuous_update=False, description="smooth",
            min=self.wIntSliderSmooth.min,
            max=self.wIntSliderSmooth.max,
            step=2, value=1,
        )

        # Checkbox to toggle the Autoscale functionality of matplotlib
        self.wCheckAutoscale = wi.Checkbox(
            description="Autoscale",
            value=True,
        )

        # Slider to select the visible y-pixel/spectra range
        self.wIntRangeSliderPixelY = wi.IntRangeSlider(
            continuous_update=False, description="y_pixels"
        )

        # Slider to select the x-pixel range used within traces
        self.wIntRangeSliderPixelX = wi.IntRangeSlider(
            continuous_update=False, description="x_pixels",
            max = PIXEL, value=(int(PIXEL*0.25), int(PIXEL*0.75)),
        )

        # Textbox to enter central wavelength of the camera in nm
        self.wTextCentralWl = wi.FloatText(
            description='central wl', value=self.central_wl,
            layout=wi.Layout(width='10%', margin='2px 50px 2px 0px')
        )

        # Dropdown menu to select x-axis calibration.
        self.wDropdownCalib = wi.Dropdown(
            description='x-axis', options=['pixel', 'nm', 'wavenumber'],
            layout=wi.Layout(width='60px', margin = "0px 130px 0px 0px"),
        )

        # Textbox to enter the wavelength of the upconversion photon
        # in nm.
        self.wTextVisWl = wi.FloatText(
            description='vis wl', value=self.vis_wl,
            layout=self.wTextCentralWl.layout
        )

        # Slider to select visible pp-delay spectrum
        self.wSliderPPDelay = wi.SelectionSlider(
            continuous_update=False, description="pp_delay",
        )

        # Slider to select range of frames used for median calculation.
        self.wRangeSliderFrame = wi.IntRangeSlider(
            continuous_update=False, description="frame"
        )

        # Checkbox to toggle the frame wise calculation of a median spectrum.
        self.wCheckFrameMedian = wi.Checkbox(
            description='median',
        )

        # Slider to select the pp_delay of the baseline.
        self.wSliderBaselinePPDelay = wi.IntSlider(
            description='pp_delay index', continuous_update=False
        )

        # Slider to select frames for median calculation.
        self.wRangeSliderBaselineFrame = wi.IntRangeSlider(
            description='frame', continuous_update=False
        )

        # Checkbox to toggle the calculation of a frame wise median of then
        # baseline.
        self.wCheckBaselineFrameMedian = wi.Checkbox(
            description='median', value=False
        )

        # Slider to select the visible baseline y-pixel/spectra
        self.wRangeSliderBaselineSpec = wi.IntRangeSlider(
            description='spectrum', continuous_update=False
        )

        # Select the x-axis of the summed plot.
        self.wDropSumAxis = wi.Dropdown(
            description='sum x-axis',
            options=('pp_delays', 'frames'),
            layout=wi.Layout(width='60px'),
        )

        # Textbox to enter an additional constant offset to the baseline.
        self.wTextBaselineOffset = wi.FloatText(
            description='Offset', value=0,
            layout=wi.Layout(widht = "10px"),
        )

        # Textbox to enter the index of the pumped spectrum.
        self.wIntTextPumped = wi.BoundedIntText(
            value=0,
            min=0,
            max=400, # Number of spectra/ypixels
            description='Pumped:'
        )

        # Textbox to enter the index of the unpumped spectrum.
        self.wIntTextUnpumped = wi.BoundedIntText(
            value=1,
            min=0,
            max=400,
            description='Unpumped:',
        )

        # Select operation to calculate bleach with.
        self.wDropdownOperator = wi.Dropdown(
            value = "-",
            options = ["-", "/"],
            description = "Operator"
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
                wi.HBox([
                    self.wSliderPPDelay,
                    self.wIntSliderSmooth,
                    self.wCheckFrameMedian
                ]),
                wi.HBox([
                    self.wRangeSliderFrame,
                    self.wIntRangeSliderPixelY,
                    self.wIntRangeSliderPixelX
                ]),
            ],
            layout=self.wVBoxData.layout
        )
        #self.wVBoxSignal.border = "1px black solid"

        self.wHBoxBleach = wi.HBox(
            [
                wi.Label("Bleach:"),
                self.wIntTextPumped,
                self.wDropdownOperator,
                self.wIntTextUnpumped,
            ],
            layout=self.wVBoxData.layout
        )


        self.wVBoxBaseline = wi.VBox(
            [
                wi.Label("Baseline:"),
                wi.HBox([
                    self.wSliderBaselinePPDelay,
                    self.wRangeSliderBaselineFrame,
                    self.wRangeSliderBaselineSpec,
                    self.wCheckBaselineFrameMedian,
                ]),
                self.wTextBaselineOffset
            ],
            layout=self.wVBoxData.layout
        )

        # List of widgets that update the figure on value change
        self._figure_widgets = [
            self.wSelectFile,
            self.wDropdownCalib,
            self.wSliderPPDelay,
            self.wRangeSliderFrame,
            self.wCheckFrameMedian,
            self.wIntRangeSliderPixelY,
            self.wIntRangeSliderPixelX,
            self.wTextVisWl,
            self.wTextCentralWl,
            self.wCheckAutoscale,
            self.wIntSliderSmooth,
            self.wToggleSubBaseline,
            self.wCheckShowBaseline,
            self.wCheckShowBleach,
            self.wCheckShowSpectra,
            self.wSliderBaselinePPDelay,
            self.wRangeSliderBaselineFrame,
            self.wRangeSliderBaselineSpec,
            self.wCheckBaselineFrameMedian,
            self.wDropSumAxis,
            self.wTextBaselineOffset,
            self.wIntTextPumped,
            self.wIntTextUnpumped,
            self.wDropdownOperator,
        ]

    def _configure_widgets(self):
        """Set all widget options. And default values."""

        # It must be save to recall this method at any time.

        self.wSliderPPDelay.options = self.data.pp_delays.tolist()
        if self.wSliderPPDelay.value not in self.wSliderPPDelay.options:
            self.wSliderPPDelay.value = self.wSliderPPDelay.options[0]
        if self.data.pp_delays.shape == (1,):
            self.wSliderPPDelay.disabled = True
        else:
            self.wSliderPPDelay.disabled = False

        self.wRangeSliderFrame.max = self.data.data.shape[FRAME_AXIS_INDEX]
        if self.data.data.shape[FRAME_AXIS_INDEX] == 1:
            self.wRangeSliderFrame.disabled = True
        else:
            self.wRangeSliderFrame.disabled = False
        if np.any(np.array(self.wRangeSliderFrame.value) >= self.wRangeSliderFrame.max):
            self.wRangeSliderFrame.value = (0, self.wRangeSliderFrame.max)

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

        self.wRangeSliderBaselineFrame.max = self.data_base.shape[FRAME_AXIS_INDEX]
        if self.wRangeSliderBaselineFrame.max is 1:
            self.wRangeSliderBaselineFrame.disabled = True
        else:
            self.wRangeSliderBaselineFrame.disabled = False
        if np.any(np.array(self.wRangeSliderBaselineFrame.value) >  self.wRangeSliderBaselineFrame.max):
            self.wRangeSliderBaselineFrame.value = 0, self.wRangeSliderBaselineFrame.max

        self.wRangeSliderBaselineSpec.max = self.data_base.shape[Y_PIXEL_INDEX]
        self.wRangeSliderBaselineSpec.min = 0
        if self.wRangeSliderBaselineSpec.max is 0:
            self.wRangeSliderBaselineSpec.disabled = True
        else:
            self.wRangeSliderBaselineSpec.disabled = False
        if self.wRangeSliderBaselineSpec.value[1] > self.wRangeSliderBaselineSpec.max:
            self.wRangeSliderBaselineSpec.value[1] = self.wRangeSliderBaselineSpec.max
            self.wRangeSliderBaselineSpec.value[0] = 0

        self.wIntTextPumped.max = self.data.data.shape[Y_PIXEL_INDEX] - 1
        self.wIntTextUnpumped.max = self.wIntTextPumped.max
        if self.wIntTextPumped.value == self.wIntTextUnpumped.value:
            self.wIntTextUnpumped.value += 1

        self._toggle_central_wl()
        self._toggle_vis_wl()
        self._toggle_sum_over()

    def _init_figure_observers(self):
        """All observers that call the *update_figure_callback* """

        # Because during widget runtime it can be necessary to stop
        # and restart the automatic figure updating to prevent flickering
        # and to speed up the gui. There is a special function to
        # set up the observers and also to remove the observers in the
        # figures_widgets list.
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
        self.wSelectBaseFile.observe(self._on_base_changed, 'value')
        self.wIntTextPumped.observe(self.y_bleach_renew, "value")
        self.wIntTextUnpumped.observe(self.y_bleach_renew, "value")
        self.wDropdownOperator.observe(self.y_bleach_renew, "value")
        self.wDropdownCalib.observe(self.x_spec_renew, "value")
        self.wTextCentralWl.observe(self.x_spec_renew, "value")
        self.wTextVisWl.observe(self.x_spec_renew, "value")
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

        # The *with* is a workaround. I need it in the test functions,
        # not the gui. Anyways, it doesn't quite work.
        with self.wSelectFile.hold_trait_notifications():
            self.wSelectFile.options = fnames
        with self.wSelectBaseFile.hold_trait_notifications():
            self.wSelectBaseFile.options = self.wSelectFile.options

    def _toggle_vis_wl(self, new=None):
        """Toggle the vis wl text box according to calibration axis.

        New: None
            Dummy keyword, so function can be uses as a callback function."""
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

        Resets all elements that need to be resetted on a baseline change."""
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
        # call it manually after a recall of _init_figure_observer
        self._unobserve_figure()
        self._configure_widgets()
        self._init_figure_observers()
        self._update_figure()

    def y_bleach_renew(self, new=None):
        """Recalculate bleach data according to gui setup."""
        self.data._bleach = self.data.get_bleach(
            self.wIntTextPumped.value,
            self.wIntTextUnpumped.value
        )
        self.data._bleach_rel = self.data.get_bleach_rel(
            self.wIntTextPumped.value,
            self.wIntTextUnpumped.value
        )

    def x_spec_renew(self, new={}):
        """Renew calibration according to gui."""
        cw = self.wTextCentralWl.value
        vis_wl= self.wTextVisWl.value
        owner = new.get("owner")
        if owner is self.wTextCentralWl and cw > 0 and vis_wl > 0:
            self.data._wavelength = self.data.get_wavelength(cw)
            self.data._wavenumber = self.data.get_wavenumber(vis_wl)
        elif owner is self.wTextVisWl and vis_wl > 0:
            self.data._wavenumber = self.data.get_wavenumber(vis_wl)
        elif owner is self.wDropdownCalib:
            if cw > 0:
                self.data._wavelength = self.data.get_wavelength(cw)
            if vis_wl > 0:
                self.data._wavenumber = self.data.get_wavenumber(vis_wl)

    def _sub_baseline(self, data):
        """Substraction of the baseline"""
        pp_delay_index = self.wSliderBaselinePPDelay.value
        frame_slice = _rangeSlider_to_slice(self.wRangeSliderBaselineFrame)
        spec_slice = _rangeSlider_to_slice(self.wRangeSliderBaselineSpec)
        num_of_spec = spec_slice.stop - spec_slice.start
        base = self.data_base
        if self.wCheckBaselineFrameMedian.value:
            base = np.median(base, FRAME_AXIS_INDEX)
        else:
            base = base[:, 0]
        if self.wIntSliderSmoothBase.value != 1:
            base = medfilt(base, (1, self.wIntSliderSmoothBase.value))
        if num_of_spec is not 1 or \
           num_of_spec is not self.data.number_of_spectra or \
           base.shape[0] is not self.data.number_of_spectra:
            warnings.warn("Cant substitute background from data, because shapes don't match.")
            return self.data.data
        if num_of_spec is 1 or \
           num_of_spec is self.data.number_of_spectra:
            base = base[spec_slice]

        self.data.base = base
        return self.data.sub_base()


        # if self.wToggleSubBaseline.value:
        #     if len(self.y_base.shape) == 1:
        #         y -= np.ones_like(y) * self.y_base
        #     elif self.y_base.shape[1] == 1:
        #         y -= np.ones_like(y) * self.y_base[:, 0]
        #     else:
        #         # Check if this is possible
        #         if self.data_base.shape[SPEC_INDEX] is not self.data.data.shape[SPEC_INDEX]:
        #             message = 'Cant subtract baseline spectra wise due to' \
        #                       'unmatched dimensions. Data shape is %s, but' \
        #                       'baseline shape is %s' %\
        #                        (self.data.data.shape, self.data_base.shape)
        #             warnings.warn(message)
        #             self.wToggleSubBaseline.value = False
        #             return y
        #         base = self.data_base[self.wSliderBaselinePPDelay.value, :]
        #         if self.wCheckBaselineFrameMedian.value:
        #             base = np.median(base[frame_slice], 0)
        #         else:
        #             base = base[frame_slice.start]
        #         y -= base[None, None, :] + self.wTextBaselineOffset.value
        # return y

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

    @property
    def x_spec(self):
        """X data of the *Signal* plot. """
        if self.wDropdownCalib.value == 'pixel':
            x = self.data.pixel
        elif self.wDropdownCalib.value == 'nm':
            x = self.data.wavelength
        elif self.wDropdownCalib.value == 'wavenumber':
            x = self.data.wavenumber
        return x

    @property
    def y_spec(self):
        """Y data of the *Signal* plot."""

        pp_delays = getattr(self.data, 'pp_delays')
        frame_slice = _rangeSlider_to_slice(self.wRangeSliderFrame)
        y_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelY)
        pp_delay_index = np.where(
            self.wSliderPPDelay.value == pp_delays)[0][0]

        ret = self.data.data[
            pp_delay_index,
            frame_slice,
            y_slice,
        ]
        if self.wCheckFrameMedian.value:
            ret = np.median(ret, FRAME_AXIS_INDEX)
        else:
            ret = ret[0]

        if self.wIntSliderSmooth.value is not 0:
            ret = medfilt(ret, (1, self.wIntSliderSmooth.value))
        # ret =  self._sub_baseline(self.data.data)[pp_delay_index, :, :, :]

        # # We don't want to see multiple frame slices
        # # Its either one specific frame, or the median.
        # if self.wCheckFrameMedian.value:
        #     ret = np.median(ret[frame_slice], FRAME_AXIS_INDEX)
        # else:
        #     ret = ret[
        #         frame_slice.start,
        #         :,
        #         :
        #     ]
        # ret = ret[y_slice, :]
        # # Must be done here, because it works only on 2d data.
        # # TODO I could use the not 2d version
        # if self.wIntSliderSmooth.value != 1:
        #     ret = medfilt2d(ret, (1, self.wIntSliderSmooth.value))
        return ret.T

    @property
    def x_base(self):
        """x data of the baseline in the *Signal* plot"""
        return self.x_spec

    @property
    def y_base(self):
        """y data of the baseline in the *Signal* plot."""

        frame_slice = _rangeSlider_to_slice(self.wRangeSliderBaselineFrame)
        spec_slice = _rangeSlider_to_slice(
            self.wRangeSliderBaselineSpec
        )
        ret = self.data_base
        if self.wCheckBaselineFrameMedian.value:
            data = self.data_base[
                self.wSliderBaselinePPDelay.value,
                frame_slice,
                spec_slice,
                :]
            y = np.median(data, 0) + self.wTextBaselineOffset.value
        else:
            y = self.data_base[
                    self.wSliderBaselinePPDelay.value,
                    frame_slice.start,
                    spec_slice,
                    :] + self.wTextBaselineOffset.value
        return y.T

    #TODO Remove this
    @property
    def x_sum(self):
        """x data of the summed plot."""
        if 'pp_delays' in self.wDropSumAxis.value:
            return self.data.pp_delays
        elif 'frames' in self.wDropSumAxis.value:
            return np.arange(self.data.frames)
        raise NotImplementedError('got %s for wDropSumAxis' % self.wDropSumAxis.value)

    #TODO split up in two methods. One for pp_delays and one for frames
    @property
    def sum_pp_delays(self):
        frame_slice = _rangeSlider_to_slice(self.wRangeSliderFrame)
        y_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelY)
        x_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelX)
        pp_delays = getattr(self.data, 'pp_delays')
        pp_delay_index = np.where(
            self.wSliderPPDelay.value == pp_delays)[0][0]

        y = self._sub_baseline(self.data.data)
        y = y[:, :, y_slice, x_slice]

        if 'pp_delays' in self.wDropSumAxis.value:
            if self.wCheckFrameMedian.value:
                y = np.median(y[:, frame_slice], FRAME_AXIS_INDEX)
            else:
                y = y[:, frame_slice.start]


    @property
    def y_sum(self):
        """y data of the summed plot."""
        frame_slice = _rangeSlider_to_slice(self.wRangeSliderFrame)
        y_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelY)
        x_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelX)
        pp_delays = getattr(self.data, 'pp_delays')
        pp_delay_index = np.where(
            self.wSliderPPDelay.value == pp_delays)[0][0]

        y = self._sub_baseline(self.data.data)
        y = y[:, :, y_slice, x_slice]

        if 'pp_delays' in self.wDropSumAxis.value:
            if self.wCheckFrameMedian.value:
                y = np.median(y[:, frame_slice], FRAME_AXIS_INDEX)
            else:
                y = y[:, frame_slice.start]

        elif 'frames' in self.wDropSumAxis.value:
            y = y[pp_delay_index]

        # The median caltulation is the most expencive calculation
        # here, to keep it as fast as possible, we do it on the
        # least amount of data possible.
        if self.wIntSliderSmooth.value is not 1:
            y = medfilt(y, (1, 1, self.wIntSliderSmooth.value))

        return y.sum(X_PIXEL_INDEX)

    @property
    def sum_x(self):
        warnings.warn("sum_x is deprecated. Use x_sum")
        return x_sum

    @property
    def sum_y(self):
        warnings.warn("sum_y is deprecated plz use y_sum")
        return self.y_sum

    @property
    def x_vlines(self):
        ret = [self.x_spec[self.wIntRangeSliderPixelX.value[0]],
               self.x_spec[self.wIntRangeSliderPixelX.value[1] - 1]]
        return ret

    @property
    def x_bleach(self):
        return self.x_spec

    @property
    def y_bleach(self):
        """The selection of the bleach for the plot"""

        y_slice_inds = self.wIntTextPumped.value, self.wIntTextUnpumped.value
        if self.wDropdownOperator.value == "-":
            data = self.data.bleach
        if self.wDropdownOperator.value == "/":
            data = self.data.bleach_rel
        # bleach data is 3d with[pp_delay, frame, x_pixel]

        pp_delays = getattr(self.data, 'pp_delays')
        pp_delay_index = np.where(
            self.wSliderPPDelay.value == pp_delays)[0][0]
        frame_slice = _rangeSlider_to_slice(self.wRangeSliderFrame)
        data = data[pp_delay_index]

        if self.wIntSliderSmooth.value != 1:
            data = medfilt(data, (1, self.wIntSliderSmooth.value))

        # TODO Baseline handling
        if self.wCheckFrameMedian.value:
            data = np.median(data[frame_slice], 0)
        else:
            dat = data[frame_slice.start]
        # TODO use the not 2d version
        return data.T


class WidgetFigures():
    """Collect figure init and update functions within this class"""
    axes_grid = np.array([[]]) # a 2d array with the figure axes

    @property
    def fig(self):
        return self._fig

    @property
    def axes(self):
        return self._fig.axes

    def _update_figure(self):
        # OVERWRITE THIS FUNCTION
        pass

    def _update_figure_callback(self, new):
        """A callback version of _update_figure for usage in observers."""
        self._update_figure()

    def init_single_figure(self):
        """Init the fiures and axes"""
        if not self._fig:
            self._fig, self.axes_grid = plt.subplots(1, 1, figsize=self._figsize, squeeze=False)
        # This allows for redrawing the axis on an already existing figure.
        elif Counter(self.axes) != Counter(self.axes_grid.flatten().tolist()) or len(self.axes) is 0:
            self._fig.set_size_inches(self._figsize, forward=True)
            self.axes_grid = np.array([[self._fig.add_subplot(111)]])

    def init_two_figures(self):
        """Init the two axis figure."""
        if not self._fig:
            self._fig, self.axes_grid = plt.subplots(1, 2, figsize=self._figsize, squeeze=False)
        # This allows for redrawing the axis on an already existing figure.
        elif Counter(self.axes) != Counter(self.axes_grid.flatten().tolist()) or len(self.axes) is 0:
            self._fig.set_size_inches(self._figsize, forward=True)
            self.axes_grid = np.array([[
                self._fig.add_subplot(121),
                self._fig.add_subplot(122)
            ]])

    def autoscale(self, ax):
        if self.wCheckAutoscale.value:
            self._ax_xlim = ax.get_xlim()
            self._ax_ylim = ax.get_ylim()
        if not isinstance(self._ax_xlim, type(None)):
            ax.set_xlim(*self._ax_xlim)
        if not isinstance(self._ax_ylim, type(None)):
            ax.set_ylim(*self._ax_ylim)

    def plot_spec(self, ax):
        if self.wCheckShowSpectra.value:
            lines = ax.plot(self.x_spec, self.y_spec)
            _set_rangeSlider_num_to_label(
                lines, self.wIntRangeSliderPixelY, "Spec"
            )

    def plot_base(self, ax):
        if self.wCheckShowBaseline.value:
            lines = ax.plot(self.x_base, self.y_base)
            _set_rangeSlider_num_to_label(
                lines, self.wRangeSliderBaselineSpec, "Baseline"
            )

    def plot_bleach(self, ax):
        if self.wCheckShowBleach.value:
            label = "Bleach%i-%i"%(self.wIntTextPumped.value,
                                 self.wIntTextUnpumped.value)
            lines = ax.plot(self.x_bleach, self.y_bleach, label=label)

    def plot_sum(self, ax):
        y_sum = self.y_sum #TODO is this needed or is this automatically cached?
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

    def redraw_figure(self):
        self._fig.canvas.draw()
        for ax in self.axes:
            ax.figure.canvas.draw()

    def _on_xlim_changed(self, new=None):
        """Callback for the *Signal* axis."""
        # Called when the xlim of the *Signal* plot is changed
        self._ax_xlim = self.axes[0].get_xlim()

    def _on_ylim_changed(self, new=None):
        # Called when the ylim of the *Signal* plot is cahnged
        self._ax_ylim = self.axes[0].get_ylim()


class SpecAndBase(WidgetBase, WidgetFigures):
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
                wi.HBox([
                    self.wDropdownCalib,
                    self.wTextCentralWl,
                    self.wTextVisWl
                ]),
        ])

    def _init_figure(self):
        """Init the fiures and axes"""
        self.init_single_figure()

    def _update_figure(self):
        """Is called on all gui element changes.

        This function renders the plot. When ever you want to make changes
        visible in the figure you need to run this."""
        self._init_figure()
        ax = self.axes[0]
        ax.clear()
        self.plot_spec(ax)
        self.plot_base(ax)
        ax.legend(framealpha=0.5)
        ax.set_xlabel(self.wDropdownCalib.value)
        ax.set_title('Spectrum')
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
        self.autoscale(ax)
        self.redraw_figure()


class SpecAndSummed(WidgetBase, WidgetFigures):
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
        self.init_two_figures()
        #TODO Axes is too small on summed

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
        """Update the figure of the gui."""
        self._init_figure()

        fontsize=8
        ax = self.axes[0]
        ax.clear()
        self.plot_spec(ax)
        self.plot_base(ax)
        ax.vlines(self.x_vlines, *ax.get_ylim(),
                  linestyle="dashed")
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        #ax.set_title("Signal")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
        ax.legend(framealpha=0.5)
        self.autoscale(ax)

        ax = self.axes[1]
        ax.clear()
        self.plot_sum(ax)
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3g'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax.yaxis.tick_right()

        self.redraw_figure()



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
            wi.HBox([self.wRangeSliderFrame, self.wIntRangeSliderPixelY, self.wIntRangeSliderPixelX, self.wCheckFrameMedian,
                  #self.wIntRangeSliderPixelX
            ]),
            wi.HBox([self.wDropdownCalib, self.wDropSumAxis]),
            self.wTextCentralWl, self.wTextVisWl,
        ])

# This is broken
class ImgView(WidgetBase):
    """A Class to view full spe images."""
    def __init__(self, *args, figsize=(8,6), **kwargs):
        super().__init__(*args, figsize=figsize, **kwargs)
        self.axes_grid = np.array([[]])

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
        frame_slice = _rangeSlider_to_slice(self.wRangeSilderFrames)
        view_data = self.data.data[self.wSliderPPDelay.value, frame_slice.start]
        ax = self.axes[0]
        plt.sca(ax)
        ax.clear()
        img = ax.imshow(view_data, interpolation=self.w_interpolate.value,
                  origin="lower", aspect="auto")
        plt.colorbar(img)

        axl = self.axes[1]
        axl.clear()
        y_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelY)
        view_data2 = self.data.data[
            self.wSliderPPDelay.value, self.wRangeSliderFrame.value[0], y_slice
        ].sum(Y_PIXEL_INDEX)
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

class Bleach(WidgetBase, WidgetFigures):
    def __init__(self, *args, figsize=(10,5), **kwargs):
        super().__init__(*args, figsize=figsize, **kwargs)
        self.axes_grid = np.array([[]])

    def _init_figure(self):
        self.init_single_figure()

    def _init_widget(self):
        import ipywidgets as wi
        super()._init_widget()
        folder_box = wi.HBox([wi.Label("Folder",
                              layout=wi.Layout(margin='0px 123px 0px 0px'))
                              , self.wTextFolder])
        self.wVBoxData = wi.VBox(
            [
                wi.Label("Data:"),
                folder_box,
                wi.HBox([
                    wi.VBox([
                        self.wToggleSubBaseline,
                        self.wCheckShowSpectra,
                        self.wCheckShowBaseline,
                        self.wCheckShowBleach,
                        self.wCheckAutoscale,
                    ]),
                    self.wSelectFile,
                    self.wSelectBaseFile,
                ]),
            ],
            layout=wi.Layout(margin = '2px 0px 16px 0px')
        )

        self.children = wi.VBox([
                self.wVBoxData,
                self.wVBoxSignal,
                self.wHBoxBleach,
                self.wVBoxBaseline,
                wi.HBox([self.wDropdownCalib, self.wTextCentralWl, self.wTextVisWl]),
                self.wDropSumAxis,
        ])

    def _update_figure(self):
        self._init_figure()
        ax = self.axes[0]
        ax.clear()
        self.plot_spec(ax)
        self.plot_base(ax)
        self.plot_bleach(ax)
        ax.legend(framealpha=0.5)
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
        self.autoscale(ax)

        self.redraw_figure()


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
        self.widgets[0]._update_figure()
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
        spec = w0._sub_baseline(w0.data.data)
        #TODO add frame_med filter here.
        ir = np.ones_like(spec) * w1.y_sepc.T

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
        if w0.y_spec.shape[0] != w1.y_spec.shape[0]:
            self.w_normalize.disable = True
            return False
        if w1.y_spec.shape[1] is 1:
            self.w_normalize.disabled = False
            return True
        if w0.y_spec.shape[1] != w1.y_spec.shape[1]:
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
    if range_value_tuple[0] != range_value_tuple[1]:
        return slice(*range_value_tuple)
    if range_value_tuple[1] != max:
        return slice(range_value_tuple[0], range_value_tuple[1]+1)
    return slice(range_value_tuple[0]-1, range_value_tuple[1])

def _rangeSlider_to_slice(rangedSlider):
    """Get a slice from a ranged slider."""
    res = _slider_range_to_slice(rangedSlider.value, rangedSlider.max)
    return res

def to_slice(attribute):
    # This can be used as a decorator, to get slices from Rangedwidgets
    # I'm currently not using it, beacuse I think its more complicated,
    # then explicitly calling the rangeSlider_to_slice function on the
    # Sliders.
    def _to_slice(f):
        def wrapper(self, *args):
            widget = getattr(self, attribute)
            return rangeSlider_to_slice(widget)
        return wrapper
    return _to_slice

def _set_rangeSlider_num_to_label(lines, rangeSlider, label_base=""):
    """Use a rangeSlider, to add rangeSlider values to label_base

    lines: The lines to set the label of.
    rangeSlider: The rangeSlider to extract values from
    label_base: base string of the label that the number is appended to."""
    spectra = rangeSlider.value
    y_slice = _rangeSlider_to_slice(rangeSlider)
    for i in range(y_slice.start, y_slice.stop):
        label = label_base + str(i)
        # - y_slice.start is needed because lines is simple list,
        # while i can start from any number
        line = lines[i - y_slice.start]
        line.set_label(label)
#### End of helper functions
