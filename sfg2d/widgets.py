"""Module for Widgets."""

import warnings, os
from glob import glob
from collections import Counter

import numpy as np
from scipy.signal import medfilt2d, medfilt
from json import dump, load
from traitlets import TraitError
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
        self._autoscale_buffer = [None, None] # Buffer to save autoscale values with.
        self._autoscale_buffer_2 = [None, None]
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
            layout = wi.Layout(width='90%'),
        )

        # Selection dialogue to select data from a list of files
        self.wSelectFile = wi.Select(
            layout = wi.Layout(width='41%'),
        )

        # Selection dialogue to select baseline data from a list of files
        self.wSelectBaseFile = wi.Select(
            layout = wi.Layout(width='42%'),
        )

        # Toggle button to toggle the subtraction of the baseline data
        self.wToggleSubBaseline = wi.ToggleButton(
            description='Sub Baseline',
        )

        # Checkbox to toggle the visibility of the baseline data
        self.wCheckShowBaseline = wi.Checkbox(
            description='Baseline',
            value=True,
        )

        # Checkbox to toggle the visibility of bleach data
        self.wCheckShowBleach = wi.Checkbox(
            description='Bleach',
            value=True,
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

        # Checkbox to toggle the Autoscale functionality of matplotlib
        self.wCheckAutoscaleSum = wi.Checkbox(
            description="Autoscale Sum",
            value=True,
        )

        # Slider to select the visible y-pixel/spectra range
        self.wIntRangeSliderPixelY = IntRangeSliderGap(
            continuous_update=True, description="y_pixels"
        )

        # Slider to select the x-pixel range used within traces
        self.wIntRangeSliderPixelX = IntRangeSliderGap(
            continuous_update=True, description="x_pixels",
            max = PIXEL, value=(int(PIXEL*0.25), int(PIXEL*0.75)),
        )

        # Textbox to enter central wavelength of the camera in nm
        self.wTextCentralWl = wi.FloatText(
            description='central wl', value=self.central_wl,
            layout=wi.Layout(
                width='180px',
            ),
        )

        # Dropdown menu to select x-axis calibration.
        self.wDropdownCalib = wi.Dropdown(
            description='x-axis', options=['pixel', 'nm', 'wavenumber'],
            layout=self.wTextCentralWl.layout,
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
        self.wRangeSliderFrame = IntRangeSliderGap(
            continuous_update=True, description="frame"
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
        self.wRangeSliderBaselineFrame = IntRangeSliderGap(
            description='frame', continuous_update=True
        )

        # Checkbox to toggle the calculation of a frame wise median of then
        # baseline.
        self.wCheckBaselineMedian = wi.Checkbox(
            description='median', value=False
        )

        # Slider to select the visible baseline y-pixel/spectra
        self.wRangeSliderBaselineSpec = IntRangeSliderGap(
            description='spectrum', continuous_update=True
        )

        # Select the x-axis of the summed plot.
        self.wDropSumAxis = wi.Dropdown(
            description='sum x-axis',
            options=('pp_delays', 'frames'),
            layout=self.wTextCentralWl.layout,
        )

        # Textbox to enter an additional constant offset to the baseline.
        self.wTextBaselineOffset = wi.FloatText(
            description='Offset', value=0,
            layout=wi.Layout(width='180px'),
        )

        # Textbox to enter the index of the pumped spectrum.
        self.wIntTextPumped = wi.BoundedIntText(
            value=0,
            min=0,
            max=400, # Number of spectra/ypixels
            description='Pumped:',
            layout=wi.Layout(width='180px'),
        )

        # Textbox to enter the index of the unpumped spectrum.
        self.wIntTextUnpumped = wi.BoundedIntText(
            value=1,
            min=0,
            max=400,
            description='Unpumped:',
            layout=self.wIntTextPumped.layout,
        )

        # Select operation to calculate bleach with.
        self.wDropdownOperator = wi.Dropdown(
            value = "-",
            options = ["-", "/"],
            description = "Operator",
            layout=self.wIntTextPumped.layout,
        )

        # Dropdown to toggle the visibility of Raw Normalized or None Spectra
        self.wDropShowSpectra = wi.Dropdown(
            options=["Raw", "Normalized", "None"],
            description='Spectra',
            value="Raw",
            layout=self.wTextCentralWl.layout,
        )

        # Dropdown to toggle the visibility of Summed Spectra
        self.wDropShowSummed = wi.Dropdown(
            options=["Raw", "Normalized", "Bleach"],
            description='Summed',
            value="Raw",
            layout=self.wTextCentralWl.layout,
        )

        ### Aligning boxers ###
        folder_box = wi.HBox([
            wi.Label("Folder"),
            self.wTextFolder,
        ])
        self.wVBoxData = wi.VBox([
            folder_box,
            wi.HBox([
                wi.Label('File'),
                self.wSelectFile,
                wi.Label(
                    'Base',
                    layout=wi.Layout(margin='0px 0px 10px 10px'),
                ),
                self.wSelectBaseFile,
            ]),
        ])

        self.wVBoxSignal = wi.VBox(
            [
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
                self.wToggleSubBaseline,
            ],
        )

        self.wHBoxBleach = wi.HBox(
            [
                self.wIntTextPumped,
                self.wDropdownOperator,
                self.wIntTextUnpumped,
            ],
        )


        self.wVBoxBaseline = wi.VBox(
            [
                wi.HBox([
                    self.wSliderBaselinePPDelay,
                    self.wRangeSliderBaselineFrame,
                    self.wRangeSliderBaselineSpec,
                    self.wCheckBaselineMedian,
                ]),
                self.wTextBaselineOffset
            ],
        )

        # Accordions from boxes
        self.wAccordionData = wi.Accordion(children = [self.wVBoxData, self.wVBoxSignal, self.wVBoxBaseline, self.wHBoxBleach])
        self.wAccordionData.set_title(0, 'Data')
        self.wAccordionData.set_title(1, 'Spectrum')
        self.wAccordionData.set_title(2, 'Baseline')
        self.wAccordionData.set_title(3, 'Bleach')

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
            self.wCheckAutoscaleSum,
            self.wIntSliderSmooth,
            self.wToggleSubBaseline,
            self.wCheckShowBaseline,
            self.wCheckShowBleach,
            self.wDropShowSpectra,
            self.wDropShowSummed,
            self.wSliderBaselinePPDelay,
            self.wRangeSliderBaselineFrame,
            self.wRangeSliderBaselineSpec,
            self.wCheckBaselineMedian,
            self.wDropSumAxis,
            self.wTextBaselineOffset,
            self.wIntTextPumped,
            self.wIntTextUnpumped,
            self.wDropdownOperator,
        ]

        # List of widgets that have a changeable state.
        # Upon saving the gui state these widgets get saved
        self._state_widgets = [self.wTextFolder, self.wSelectBaseFile] + self._figure_widgets

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
        self.wDropdownCalib.observe(self.x_spec_renew, "value")
        self.wTextCentralWl.observe(self.x_spec_renew, "value")
        self.wTextVisWl.observe(self.x_spec_renew, "value")
        self.wToggleSubBaseline.observe(self.on_subBaseline_toggled, 'value')
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
        if self.data.number_of_frames is 1:
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
        `WidgetBase.wTextFolder.value` and `WidgetBase.wSelectFile.value`.
        The `WidgetBase._central_wl` property gets reseted, and child
        widget elements, like e.g. WidgetBase.wSliderPPDelay are checked for
        correctness and reseted."""
        fname = self.wTextFolder.value + "/" + self.wSelectFile.value
        self.data = SfgRecord(fname)
        self._central_wl = None
        # Deactivating the observers here prevents flickering
        # and unneeded calls of _update_figure. Thus we
        # call it manually after a recall of _init_observer
        try:
            # Try needed to allow for double call of this function.
            # TODO This is hacky, Maybe I can get around this
            # by using the new dict.
            self._unobserve_figure()
            keep_figure_unobserved = False
        except ValueError:
            keep_figure_unobserved = True
        self._configure_widgets()
        self.on_subBaseline_toggled()
        if not keep_figure_unobserved:
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
        try:
            # Try needed to allow for double call of this function.
            # TODO This is hacky, Maybe I can get around this
            # by using the new dict.
            keep_figure_unobserved = False
            self._unobserve_figure()
        except ValueError:
            keep_figure_unobserved = True
        self._configure_widgets()
        self.on_subBaseline_toggled()
        if not keep_figure_unobserved:
            self._init_figure_observers()
            self._update_figure()

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

    #TODO restet to set_baseline
    def on_subBaseline_toggled(self, new={}):
        """Callback function for the baseline toggle."""
        if self.wToggleSubBaseline.value:
            self._sub_baseline()
        else:
            self.data.data = self.data.rawData

    def _sub_baseline(self):
        # Keep old baseline if new is invalid.
        try:
            base = self.y_base.T
            self.data.base = base
        except ValueError:
            pass
        return self.data.sub_base()

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

        if self.wDropShowSpectra.value == "Normalized":
            ret = self.data.get_normalized(use_rawData=True)

        elif self.wToggleSubBaseline.value:
            # Keep the old baseline if we cant set it.
            try:
                self.data.base = self.y_base.T
            except ValueError:
                pass
            ret = self.data.get_baselinesubed()

        else:
            ret = self.data.rawData

        ret = ret[
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
        if self.wCheckBaselineMedian.value:
            y = self.data_base[
                :,
                frame_slice,
                spec_slice,
                :]
            y = np.median(y, 0)
            y = np.median(y, 0)
        else:
            y = self.data_base[
                    self.wSliderBaselinePPDelay.value,
                    frame_slice.start,
                    spec_slice,
                    :]
        y = y + self.wTextBaselineOffset.value
        return y.T

    #TODO Remove this
    @property
    def x_sum(self):
        """x data of the summed plot."""
        if 'pp_delays' in self.wDropSumAxis.value:
            return self.data.pp_delays
        elif 'frames' in self.wDropSumAxis.value:
            return np.arange(self.data.number_of_frames)
        raise NotImplementedError('got %s for wDropSumAxis' % self.wDropSumAxis.value)

    @property
    def sum_pp_delays(self):
        """Returns the pp_delay wise sum off the baseline subed data."""
        frame_slice = _rangeSlider_to_slice(self.wRangeSliderFrame)
        y_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelY)
        x_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelX)
        frame_median = self.wCheckFrameMedian.value
        medfilt_kernel = (1, 1, 1, self.wIntSliderSmooth.value)
        ret = self.data.get_trace_pp_delay(
            frame_slice,
            y_slice,
            x_slice,
            frame_median,
            medfilt_kernel,
        )
        # Pics the first selected frame
        if not frame_median:
            ret = ret[:, 0]
        return ret

    @property
    def sum_frames(self):
        """Summes of given pixel range frame wise."""
        y_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelY)
        x_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelX)
        pp_delays = getattr(self.data, 'pp_delays')
        pp_delay_index = np.where(
            self.wSliderPPDelay.value == pp_delays)[0][0]

        if self.wToggleSubBaseline.value:
            y = self._sub_baseline()
        else:
            y = self.data.data
        y = y[pp_delay_index, :, y_slice, x_slice]

        if self.wIntSliderSmooth.value is not 1:
            y = medfilt(y, (1, 1, self.wIntSliderSmooth.value))

        y = np.sum(y, -1)
        return y

    @property
    def y_sum(self):
        """y data of the summed plot."""
        if 'pp_delays' in self.wDropSumAxis.value:
            if 'Raw' in self.wDropShowSummed.value:
                self.data.data = self.data.get_baselinesubed(use_rawData=True)
                y = self.sum_pp_delays
            elif 'Normalized' in self.wDropShowSummed.value:
                self.data.data = self.data.get_normalized(use_rawData=True)
                y = self.sum_pp_delays
            elif 'Bleach' in self.wDropShowSummed.value:
                y = self.y_sum_bleach
        elif 'frames' in self.wDropSumAxis.value:
            y = self.sum_frames
        return y

    @property
    def sum_x(self):
        warnings.warn("sum_x is deprecated. Use x_sum")
        return self.x_sum

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
            data = self.data.get_bleach(
                self.wIntTextPumped.value,
                self.wIntTextUnpumped.value
            )
        elif self.wDropdownOperator.value == "/":
            data = self.data.get_bleach_rel(
                self.wIntTextPumped.value,
                self.wIntTextUnpumped.value
            )
        # bleach data is 3d with[pp_delay, frame, x_pixel]

        pp_delays = getattr(self.data, 'pp_delays')
        pp_delay_index = np.where(
            self.wSliderPPDelay.value == pp_delays)[0][0]
        frame_slice = _rangeSlider_to_slice(self.wRangeSliderFrame)
        data = data[pp_delay_index]

        # TODO Baseline handling
        if self.wCheckFrameMedian.value:
            data = np.median(data[frame_slice], 0)
        else:
            data = data[frame_slice.start]

        if self.wIntSliderSmooth.value != 1:
            data = medfilt(data, self.wIntSliderSmooth.value)

        return data.T

    @property
    def y_sum_bleach(self):
        frame_slice = _rangeSlider_to_slice(self.wRangeSliderFrame)
        x_slice = _rangeSlider_to_slice(self.wIntRangeSliderPixelX)

        if self.wDropdownOperator.value == "-":
            data = self.data.bleach
        elif self.wDropdownOperator.value == "/":
            data = self.data.bleach_rel

        if self.wCheckFrameMedian.value:
            data = np.median(data[:, frame_slice], 1)
        else:
            data = data[:, frame_slice.start]

        if self.wIntSliderSmooth.value != 1:
            data = medfilt(data, (1, self.wIntSliderSmooth.value))

        data = np.sum(data[:,x_slice], -1)
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


    def plot_spec(self, ax):
        if self.wDropShowSpectra.value != "None":
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

    def _on_ax0_lim_changed(self, new=None):
        """Callback for the *Signal* axis."""
        # Called when the xlim of the `Signal` plot is changed
        self._autoscale_buffer = _lims2buffer(self.axes[0])

    def _on_ax1_lim_changed(self, new=None):
        # Called when the ylim of the `Signal` plot is cahnged
        self._autoscale_buffer_2 = _lims2buffer(self.axes[1])

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
        visible in the figure you need to call this."""
        self._init_figure()
        ax = self.axes[0]
        ax.clear()
        self.plot_spec(ax)
        self.plot_base(ax)
        ax.legend(framealpha=0.5)
        ax.set_xlabel(self.wDropdownCalib.value)
        ax.set_title('Spectrum')
        ax.callbacks.connect('xlim_changed', self._on_ax0_lim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ax0_lim_changed)
        if self.wCheckAutoscale.value:
            self._autoscale_buffer_2 = _lims2buffer(ax)
        else:
            _buffer2lims(ax, self._autoscale_buffer_2)
        self.redraw_figure()


# TODO Find Better name
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
                self.wAccordionData,
                wi.HBox([
                    self.wDropdownCalib,
                    self.wTextCentralWl,
                    self.wTextVisWl,
                    self.wCheckShowBaseline,
                ]),
                wi.HBox([
                    self.wDropSumAxis,
                    self.wDropShowSpectra,
                    self.wDropShowSummed,
                    self.wCheckShowBleach,
                    self.wCheckAutoscale,
                    self.wCheckAutoscaleSum,
                ])
        ])

    def _update_figure(self):
        """Update the figure of the gui."""
        self._init_figure()

        fontsize=8
        ax = self.axes[0]
        ax.clear()
        self.plot_spec(ax)
        self.plot_base(ax)
        self.plot_bleach(ax)
        ax.vlines(self.x_vlines, *ax.get_ylim(),
                  linestyle="dashed")
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        #ax.set_title("Signal")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax.callbacks.connect('xlim_changed', self._on_ax0_lim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ax0_lim_changed)
        if self.wCheckAutoscale.value:
            self._autoscale_buffer = _lims2buffer(ax)
        else:
            _buffer2lims(ax, self._autoscale_buffer)
        ax.legend(framealpha=0.5)

        ax = self.axes[1]
        ax.clear()
        self.plot_sum(ax)
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3g'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax.yaxis.tick_right()
        ax.callbacks.connect('xlim_changed', self._on_ax1_lim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ax1_lim_changed)
        if self.wCheckAutoscaleSum.value:
            self._autoscale_buffer_2 = _lims2buffer(ax)
        else:
            _buffer2lims(ax, self._autoscale_buffer_2)

        self.redraw_figure()

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
        #TODO rename widgets to tabs
        for widget in args:
            widget._configure_widgets()
            children.append(widget.children)
            widget._fig = self.wi_fig

        self.w_tabs = wi.Tab(children=children)
        self.children = self.w_tabs
        self.wButtonNormalize = wi.Button(description='Normalize')
        self.wButtonSaveGui = wi.Button(description='Save Gui')
        self.wButtonLoadGui = wi.Button(description='Load Gui')
        self.children = wi.VBox([
            self.w_tabs,
            wi.HBox([
                self.wButtonNormalize,
                self.wButtonSaveGui,
                self.wButtonLoadGui,
            ])
        ])

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
        self.wButtonNormalize.on_click(self._on_normalize)

        # observers to w? must be re initiated on each data change.
        w0, w1, *_ = self.widgets
        # Need to make shure, that Normalize button is only clickable,
        # When the shape of the data allows for normalization
        w0.wIntRangeSliderPixelY.observe(self._is_normalizable_callback, "value")
        w1.wIntRangeSliderPixelY.observe(self._is_normalizable_callback, "value")
        self.wButtonSaveGui.on_click(self._on_save_gui_clicked)
        self.wButtonLoadGui.on_click(self._on_load_gui_clicked)

    def _on_tab_changed(self, new):
        if debug:
            print("Dashboard._on_tab_changed called")
        axes = self.wi_fig.axes
        for ax in axes:
             self.wi_fig.delaxes(ax)
        page = self.w_tabs.selected_index
        widget = self.widgets[page]
        widget._update_figure()

    def _on_normalize(self, new):
        if debug:
            print("Normalize._on_normalize called.")
        w0, w1, *_ = self.widgets
        w0.data.norm = w1.y_spec.T

    def _on_save_gui_clicked(self, new):
        """Save gui status to a json text file.

        Each tab of the dashboard gets a separate list entry. Each widget value
        is saved as an element of a list of a list."""
        save_file = self.widgets[0].wTextFolder.value + '/.last_state.json'
        with open(save_file, 'w') as outfile:
            save_list = []
            for i in range(len(self.widgets[:2])):
                w = self.widgets[i]
                save_list.append([widget.value for widget in w._state_widgets]),
            dump(save_list, outfile)

    def _on_load_gui_clicked(self, new):
        folder = self.widgets[0].wTextFolder.value
        with open(folder + '/.last_state.json', 'r') as infile:
            imp = load(infile)
            for i in range(len(self.widgets[:2])):
                w = self.widgets[i]
                w._unobserve_figure()
                for j in range(len(w._state_widgets)):
                    widget = w._state_widgets[j]
                    try:
                        widget.value = imp[i][j]
                        # We need to call the folder manually. Otherwise changes
                        # are not recognized.
                        if j == 0:
                            w._on_folder_submit(None)
                    except TraitError:
                        msg = "Count load value %s of widget %s"%(imp[i][j],
                                                                 widget)
                        print(msg)
                        break
                w._init_figure_observers()
        self._on_tab_changed(None)

    @property
    def _is_normalizable(self):
        w0, w1, *_  = self.widgets
        if w0.y_spec.shape[0] != w1.y_spec.shape[0]:
            self.wButtonNormalize.disable = True
            return False
        if w1.y_spec.shape[1] is 1:
            self.wButtonNormalize.disabled = False
            return True
        if w0.y_spec.shape[1] != w1.y_spec.shape[1]:
            self.wButtonNormalize.disabled = True
            return False
        self.wButtonNormalize.disabled = False
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
    return slice(*rangedSlider.value)

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

def _lims2buffer(ax):
    """Set buffer values according to axis"""
    buffer = [None, None]
    buffer[0] = list(ax.get_xlim())
    buffer[1] = list(ax.get_ylim())
    return buffer

def _buffer2lims(ax, buffer):
    if not isinstance(buffer[0], type(None)):
        ax.set_xlim(*buffer[0])
    if not isinstance(buffer[1], type(None)):
        ax.set_ylim(*buffer[1])

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

from traitlets import validate
from ipywidgets import IntRangeSlider

class IntRangeSliderGap(IntRangeSlider):
    @validate('value')
    def enforce_gap(self, proposal):
        gap=1
        min, max = proposal.value
        oldmin, oldmax = self.value

        if min == self.max:
            min -= 1

        if (max-min) < gap:
            if oldmin == min:
                # max changed
                max = min + gap
            else:
                min = max - gap
        return (min, max)
#### End of helper functions


