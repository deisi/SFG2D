import os
from glob import glob
from collections import Counter

import numpy as np
from scipy.signal import medfilt
from json import dump, load
from traitlets import TraitError
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from traitlets import validate
from ipywidgets import IntRangeSlider

from .core import SfgRecord, concatenate_list_of_SfgRecords
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
    *WidgetBase._conf_widget_with_data* function.
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

        # Internal objects
        # Figure to draw on
        self._fig = fig
        # Central wavelength of the camera
        self._central_wl = central_wl
        # Visible wavelength for wavenumber calculation
        self._vis_wl = vis_wl
        # Size of the figure
        self._figsize = figsize
        # List of widgets that update the figure
        self._figure_widgets = []
        # Buffer to save autoscale values with.
        self._autoscale_buffer = [None, None]
        self._autoscale_buffer_2 = [None, None]
        # Buffer to save x_rois upon switching data.
        self._rois_x_pixel_buffer = [slice(None, None)]
        # Buffer unpumped and pumped data throughout switching data files
        self._unpumped_index_buffer = 0
        self._pumped_index_buffer = 1
        # List of widgets to display
        self.children = []

        # Setup all widgets
        self._init_widget()

    def __call__(self):
        """Use call to actually Render the widgets on the notebook."""
        from IPython.display import display
        self._conf_widget_with_data()
        self._init_observer()
        self._init_figure_observers()
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

        # ## Any widget that we need at some point can be added here.

        # Widget to enter a folder path as string
        self.wTextFolder = wi.Text(
            layout=wi.Layout(width='90%'),
        )

        # Selection dialogue to select data from a list of files
        self.wSelectFile = wi.SelectMultiple(
            layout=wi.Layout(width='41%'),
        )

        # Checkbox to toggle the visibility of the baseline data
        self.wCheckShowBase = wi.Checkbox(
            description='Baseline',
            value=False,
        )

        # Checkbox to toggel the visiblitiy of the norm
        self.wCheckShowNorm = wi.Checkbox(
            description='Norm',
            value=False
        )

        # Checkbox to toggle visibility of bleach
        self.wCheckShowBleachAbs = wi.Checkbox(
            description='Abs',
            value=False,
        )

        # Checkbox to toggle visibility of bleach
        self.wCheckShowBleachAbsNorm = wi.Checkbox(
            description='Abs Norm',
            value=False,
        )

        # Checkbox to toggle visibility of bleach
        self.wCheckShowBleachRel = wi.Checkbox(
            description='Rel',
            value=False,
        )

        # Checkbox to toggle visibility of bleach
        self.wCheckShowBleachRelNorm = wi.Checkbox(
            description='Rel Norm',
            value=False,
        )

        self.wCheckShowTracesBleachAbs = wi.Checkbox(
            description='Bleach Abs',
            value=False,
        )

        self.wCheckShowTracesBleachAbsNorm = wi.Checkbox(
            description='Bleach Abs Norm',
            value=False,
        )

        self.wCheckShowTracesBleachRel = wi.Checkbox(
            description='Bleach Rel',
            value=False,
        )

        self.wCheckShowTracesBleachRelNorm = wi.Checkbox(
            description='Bleach Rel Norm',
            value=False,
        )

        self.wCheckShowTracesRawData = wi.Checkbox(
            description='Raw',
            value=True,
        )

        self.wCheckShowTracesBasesubed = wi.Checkbox(
            description='Basesubed',
            value=False,
        )

        self.wCheckShowTracesNormalized = wi.Checkbox(
            description='Normalized',
            value=False,
        )

        # Region slice to select index for zero_time_subtraction
        self.wRangeZeroTime = IntRangeSliderGap(
            description="Zero Time",
            value=(0, 1), continuous_update=False,
        )

        # Snap pixel roi.
        self.wSnapXRoi = wi.Button(
            description="Snap X Region"
        )

        # Checkbox to toggle the zero_time suntraction of bleach data
        self.wCheckShowZeroTimeSubtraction = wi.Checkbox(
            description='Sub Zero Time',
            value=False,
        )

        # Slider to select the width of the smoothing kernel
        self.wIntSliderSmooth = wi.IntSlider(
            continuous_update=False, description="Smooth",
            min=1, max=19, step=2,
        )

        # Slider to select smoothing of baseline
        # TODO Is not used yet
        self.wIntSliderSmoothBase = wi.IntSlider(
            continuous_update=False, description="Smooth",
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
        self.wCheckAutoscaleTrace = wi.Checkbox(
            description="Autoscale Trace",
            value=True,
        )

        # Slider to select the visible y-pixel/spectra range
        self.wRangeSliderPixelY = IntRangeSliderGap(
            continuous_update=False, description="Spectra Region"
        )

        # Slider to select spectra step size.
        self.wIntTextPixelYStep = wi.BoundedIntText(
            description='Spectra Stepsize', value=1, min=1,
            layout=wi.Layout(width='180px',),
        )

        self.wCheckSpectraMean = wi.Checkbox(
            description='Spectra Mean',
            value=False
        )

        self.wDropdownSpectraMode = wi.Dropdown(
            description='Spectra Mode',
            options=['Index', 'Region'],
            value='Region',
            layout=wi.Layout(width='180px',),
        )

        # Slider to select the overall visible x-pixel range
        self.wRangeSliderPixelX = IntRangeSliderGap(
            continuous_update=False, description="X Region",
            max=PIXEL, value=(0, PIXEL),
        )

        # Slider to select the x-pixel range used within traces
        self.wRangeSliderTracePixelX = IntRangeSliderGap(
            continuous_update=False, description="X Trace",
            max=PIXEL, value=(int(PIXEL*0.40), int(PIXEL*0.6)),
        )

        # Textbox to enter central wavelength of the camera in nm
        self.wTextCentralWl = wi.FloatText(
            description='Central Wl', value=self.central_wl,
            layout=wi.Layout(
                width='180px',
            ),
        )

        # Dropdown menu to select x-axis calibration.
        self.wDropdownCalib = wi.Dropdown(
            description='x-axis', options=['pixel', 'wavelength', 'wavenumber'],
            layout=self.wTextCentralWl.layout,
        )

        # Textbox to enter the wavelength of the upconversion photon
        # in nm.
        self.wTextVisWl = wi.FloatText(
            description='Vis Wl', value=self.vis_wl,
            layout=self.wTextCentralWl.layout
        )

        # Slider to select visible pp-delay spectrum
        self.wSliderPPDelay = wi.IntSlider(
            description="Delay Index", continuous_update=False,
        )
        self.wRangeSliderPPDelay = IntRangeSliderGap(
            continuous_update=False, description="Delay Region",
        )
        self.wCheckDelayMedian = wi.Checkbox(
            description='Delay Median', value=False, disabled=False
        )

        # Dropdown to choose how Baseline or IR data gets send.
        self.wDropdownDelayMode = wi.Dropdown(
            description="Delay Mode", value="Index",
            options=["Index", "Region"],
            layout=wi.Layout(width='180px',)
        )

        # Slider to select range of frames used for median calculation.
        self.wSliderFrame = wi.IntSlider(
            description='Frame Index', continuous_update=False
        )

        self.wRangeSliderFrame = IntRangeSliderGap(
            continuous_update=False, description="Frame Region"
        )

        # Checkbox to toggle the frame wise calculation of a median spectrum.
        self.wCheckFrameMedian = wi.Checkbox(
            description='Frame Median',
        )

        # Dropdown to choos how Baseline and IR data gest send
        self.wDropdownFrameMode = wi.Dropdown(
            description="Frame Mode", value="Index",
            options=["Index", "Region"],
            layout=wi.Layout(width='180px',)
        )

        # Slider to select frames for median calculation.
        self.wSliderFrame = wi.IntSlider(
            description='Frame', continuous_update=False
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
            max=400,  # Number of spectra/ypixels
            description='Pumped',
            layout=wi.Layout(width='180px'),
        )

        # Textbox to enter the index of the unpumped spectrum.
        self.wIntTextUnpumped = wi.BoundedIntText(
            value=1,
            min=0,
            max=400,
            description='Unpumped',
            layout=self.wIntTextPumped.layout,
        )

        # Checkbox to toggle visibility of Raw Spectra.
        self.wCheckShowRawData = wi.Checkbox(
            description='RawData',
            value=True,
        )

        # Checkbox to toggle visibility of Basesubed.
        self.wCheckShowBasesubed = wi.Checkbox(
            description='Basesubed',
            value=False,
        )

        # Checkbox to toggle visibility of Normalized.
        self.wCheckShowNormalized = wi.Checkbox(
            description='Normalized',
            value=False,
        )

        # Dropdown to toggle view of the summed spectra
        self.wDropShowTrace = wi.Dropdown(
            options=["Raw", "Normalized", "Bleach"],
            description='Trace',
            value="Raw",
            layout=self.wTextCentralWl.layout,
        )

        self.wTextSaveRecord = wi.Text(
            description="File"
        )

        self.wButtonSaveRecord = wi.Button(
            description="Save Record"
        )

        # ### Aligning boxers ###
        self._data_box = wi.VBox([
            wi.HBox([
                wi.Label("Folder"),
                self.wTextFolder,
               ]),
            wi.HBox([
                wi.Label('File'),
                self.wSelectFile,
            ]),
        ])
        self._signal_box = wi.VBox([
            wi.HBox([
                self.wSliderPPDelay,
                self.wCheckDelayMedian,
                self.wRangeSliderPPDelay,
                self.wDropdownDelayMode,
            ]),
            wi.HBox([
                self.wSliderFrame,
                self.wCheckFrameMedian,
                self.wRangeSliderFrame,
                self.wDropdownFrameMode,
            ]),
            wi.HBox([
                self.wRangeSliderPixelY,
                self.wCheckSpectraMean,
                self.wIntTextPixelYStep,
                self.wRangeSliderPixelX,
            ]),
            wi.HBox([
                self.wIntSliderSmooth,
                self.wRangeSliderTracePixelX,
            ])
        ])
        self._calib_box = wi.HBox([
            self.wDropdownCalib,
            self.wTextCentralWl,
            self.wTextVisWl,
            self.wCheckAutoscale,
            self.wCheckAutoscaleTrace,
        ])
        self._save_record_box = wi.HBox([
            self.wTextSaveRecord,
            self.wButtonSaveRecord,
        ])

        # List of widgets that update the figure on value change
        self._figure_widgets = [
            self.wSelectFile,
            self.wSliderPPDelay,
            self.wRangeSliderPPDelay,
            self.wSliderFrame,
            self.wCheckDelayMedian,
            self.wRangeSliderFrame,
            self.wRangeSliderPixelY,
            self.wIntTextPixelYStep,
            self.wRangeSliderPixelX,
            self.wRangeSliderTracePixelX,
            self.wCheckFrameMedian,
            self.wTextVisWl,
            self.wTextCentralWl,
            self.wCheckAutoscale,
            self.wDropdownCalib,
            self.wCheckAutoscaleTrace,
            self.wCheckShowNorm,
            self.wIntSliderSmooth,
            self.wCheckShowBase,
            self.wCheckShowBleachAbs,
            self.wCheckShowBleachAbsNorm,
            self.wCheckShowBleachRel,
            self.wCheckShowBleachRelNorm,
            self.wCheckShowRawData,
            self.wCheckShowBasesubed,
            self.wCheckShowNormalized,
            self.wCheckShowTracesBleachAbs,
            self.wCheckShowTracesBleachAbsNorm,
            self.wCheckShowTracesBleachRel,
            self.wCheckShowTracesBleachRelNorm,
            self.wCheckShowTracesRawData,
            self.wCheckShowTracesBasesubed,
            self.wCheckShowTracesNormalized,
            self.wCheckSpectraMean,
            self.wDropdownSpectraMode,
            self.wDropdownDelayMode,
            self.wDropdownFrameMode,
            self.wDropShowTrace,
            self.wTextBaselineOffset,
            self.wIntTextPumped,
            self.wIntTextUnpumped,
            self.wCheckShowZeroTimeSubtraction,
            self.wRangeZeroTime,
        ]

        # Upon saving the gui state these widgets get saved
        self._save_widgets = {
            'folder': self.wTextFolder,
            'file': self.wSelectFile,
            'showBaseline': self.wCheckShowBase,
            'checkDelayMedian': self.wCheckDelayMedian,
            'showBleachAbs': self.wCheckShowBleachAbs,
            'showBleachAbsNorm': self.wCheckShowBleachAbsNorm,
            'showBleachRel': self.wCheckShowBleachRel,
            'showBleachRelNorm': self.wCheckShowBleachRelNorm,
            'showTracesBleachAbs': self.wCheckShowTracesBleachAbs,
            'showTraceBleachAbsNorm': self.wCheckShowTracesBleachAbsNorm,
            'showTracesBleachRel': self.wCheckShowTracesBleachRel,
            'showTracesBleachRelNorm': self.wCheckShowTracesBleachRelNorm,
            'showTracesRawData': self.wCheckShowTracesRawData,
            'showTracesBasesubed': self.wCheckShowTracesBasesubed,
            'showTracesNormalized': self.wCheckShowTracesNormalized,
            'delayMode': self.wDropdownDelayMode,
            'frameMode': self.wDropdownFrameMode,
            'smoothSlider': self.wIntSliderSmooth,
            'smoothBase': self.wIntSliderSmoothBase,
            'autoscale': self.wCheckAutoscale,
            'autoscaleTrace': self.wCheckAutoscaleTrace,
            'pixelY': self.wRangeSliderPixelY,
            'pixelY_step': self.wIntTextPixelYStep,
            'pixelXpixel': self.wRangeSliderPixelX,
            'tracePixelX': self.wRangeSliderTracePixelX,
            'centralWl': self.wTextCentralWl,
            'calib': self.wDropdownCalib,
            'visWl': self.wTextVisWl,
            'showNorm': self.wCheckShowNorm,
            'pp_delay_slice': self.wRangeSliderPPDelay,
            'frame_region': self.wRangeSliderFrame,
            'frame_index': self.wSliderFrame,
            'frameMedian': self.wCheckFrameMedian,
            'frame': self.wSliderFrame,
            'baselineOffset': self.wTextBaselineOffset,
            'pumped': self.wIntTextPumped,
            'unpumped': self.wIntTextUnpumped,
            'bleachZeroTimeSubtraction': self.wCheckShowZeroTimeSubtraction,
            'showTrace': self.wDropShowTrace,
            'spectraMean': self.wCheckSpectraMean,
            'spectraMode': self.wDropdownSpectraMode,
            'showRawData': self.wCheckShowRawData,
            'showBasesubed': self.wCheckShowBasesubed,
            'showNormalized': self.wCheckShowNormalized,
            'zeroTimeSelec': self.wRangeZeroTime,
        }

    def _conf_widget_with_data(self):
        """Set all widget options and default values according to data.

        This uses the data to set the state of the widget. Thus one calles
        it usually after loading new data. During operation of the widget
        this is usually not called, because then the widget updates the data.
        """
        def _set_range_slider_options(slider, record_data_index):
            """Set options of a gaped range slider.
            slider: The slider to set the options of,
            record_data_index: Index position of the property to set.
            """

            slider.max = self.data.rawData.shape[record_data_index]
            if self.data.data.shape[record_data_index] == 1:
                slider.disabled = True
            else:
                slider.disabled = False
            if np.any(np.array(slider.value) >=
                      slider.max):
                slider.value = (0, slider.max)

        def _set_int_slider_options(slider, record_data_index):
            """Set options of a slider.
            slider: The slider to set the options of,
            record_data_index: Index position of the property to set.
            """

            slider.max = self.data.rawData.shape[
                record_data_index
            ] - 1
            if slider.value > self.wSliderPPDelay.max:
                slider.value = self.wSliderPPDelay.max
            if slider.max == 1:
                slider.disabled = True
            else:
                slider.disabled = False

        # TODO Maybe I should split this up in conf_widget_options
        # and conf_widget_values.
        _set_range_slider_options(self.wRangeSliderPPDelay, PP_INDEX)
        _set_range_slider_options(self.wRangeSliderFrame, FRAME_AXIS_INDEX)
        _set_range_slider_options(self.wRangeSliderPixelY, Y_PIXEL_INDEX)
        self.wIntTextPixelYStep.max = self.wRangeSliderPixelY.max
        if self.wIntTextPixelYStep.value > self.wIntTextPixelYStep.max:
            self.wIntTextPixelYStep.value = self.wIntTextPixelYStep.max
        _set_range_slider_options(self.wRangeSliderTracePixelX, X_PIXEL_INDEX)
        _set_int_slider_options(self.wSliderPPDelay, PP_INDEX)
        _set_int_slider_options(self.wSliderFrame, FRAME_AXIS_INDEX)
        _set_range_slider_options(self.wRangeZeroTime, PP_INDEX)

        if isinstance(self.central_wl, type(None)):
            self.wTextCentralWl.value = 0
        else:
            self.wTextCentralWl.value = self.central_wl

        self.wTextVisWl.value = self.vis_wl

        # Currently not used.
        self.wSliderFrame.max = self.data.base.shape[
            FRAME_AXIS_INDEX
        ] - 1
        if self.wSliderFrame.max == 1:
            self.wSliderFrame.disabled = True
        else:
            self.wSliderFrame.disabled = False
        if self.wSliderFrame.value > self.wSliderFrame.max:
            self.wSliderFrame.value = self.wSliderFrame.max

        self.wIntTextPumped.max = self.data.data.shape[Y_PIXEL_INDEX] - 1
        self.wIntTextUnpumped.max = self.wIntTextPumped.max
        self.wIntTextUnpumped.value = self.data.unpumped_index
        self.wIntTextPumped.value = self.data.pumped_index
        if self.wIntTextPumped.value == self.wIntTextUnpumped.value:
            self.wIntTextUnpumped.value += 1

        self.wTextBaselineOffset.value = self.data.baseline_offset

        self._toggle_central_wl()
        self._toggle_vis_wl()

    def _init_figure_observers(self):
        """All observers that call the *update_figure_callback* """

        # Because during widget runtime it can be necessary to stop
        # and restart the automatic figure updating to prevent flickering
        # and to speed up the gui. There is a special function to
        # set up the observers and also to remove the observers in the
        # figures_widgets list.
        for widget in self._figure_widgets:
            widget.observe(self._update_figure_callback, "value")

    def _unobserve_figure(self):
        """Unobserver figure observers."""
        for widget in self._figure_widgets:
            try:
                widget.unobserve(self._update_figure_callback, 'value')
            except ValueError:
                if debug:
                    print('Cant unobserve {} description is {}'.format(
                        widget,                                                                        widget.description
                    ))

    def _init_observer(self):
        """Set all observer of all subwidgets."""
        # This registers the callback functions to the gui elements.
        # After a call of _init_observer, the gui elements start to
        # actually do something, namely what ever is defined within the
        # callback function of the observer.
        self.wTextFolder.on_submit(self._on_folder_submit)
        self.wSelectFile.observe(self._load_data, 'value')
        self.wDropdownCalib.observe(self._on_calib_changed, "value")
        self.wTextCentralWl.observe(self.x_spec_renew, "value")
        self.wTextVisWl.observe(self.x_spec_renew, "value")
        self.wIntTextPumped.observe(self._on_pumped_index_changed, "value")
        self.wIntTextUnpumped.observe(self._on_unpumped_index_changed, "value")
        self.wCheckDelayMedian.observe(self._on_delay_median_clicked, "value")
        self.wDropdownDelayMode.observe(self._on_delay_mode_changed, "value")
        self.wCheckFrameMedian.observe(self._on_frame_median_clicked, "value")
        self.wDropdownFrameMode.observe(self._on_frame_mode_changed, "value")
        self.wRangeSliderTracePixelX.observe(self._set_roi_trace_x_pixel,
                                        "value")
        self.wRangeSliderFrame.observe(self._set_roi_frames,
                                       "value")

        self.wRangeSliderPixelY.observe(self._set_roi_spectra,
                                        "value")
        self.wRangeSliderPPDelay.observe(self._set_roi_delays,
                                         "value")
        self.wButtonSaveRecord.on_click(self._on_save_record)
        self.wCheckShowZeroTimeSubtraction.observe(
            self._set_zero_time_subtraction, "value"
        )
        self.wTextBaselineOffset.observe(
            self._on_baseline_offset_changed, "value"
        )
        self.wRangeZeroTime.observe(self._set_zero_time_selec, "value")
        self.wSnapXRoi.on_click(self._snap_x_roi)
        #self._init_figure_observers()

    def _on_folder_submit(self, new=None):
        """Called when folder is changed."""
        if not os.path.isdir(self.wTextFolder.value):
            print('Warning folder {} not found'.format(self.wTextFolder.value))
            return

        if debug:
            print("_on_folder_submit_called")
        if debug > 1:
            print("fnames:", self.fnames)

        # The *with* is a workaround. I need it in the test functions,
        # not the gui. Anyways, it doesn't quite work.
        with self.wSelectFile.hold_trait_notifications():
            self.wSelectFile.options = self.fnames

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

        The new keyword exists, so it can also server as a callback function.
        """
        if self.wDropdownCalib.value == 'pixel' or self.data._type == 'spe':
            self.wTextCentralWl.disabled = True
        else:
            self.wTextCentralWl.disabled = False

    def _load_data(self, new=None):
        """Update the internal data objects.

        Loads data from hdd, and sets data properties according to gui settings.
        Afterwards, the gui settings are configured agains the data again to
        ensure consistency.

        Sheme:
        load data ---> update data  ---> configure widget options and values
        """

        if len(self.wSelectFile.value) == 0:
            return
        elif len(self.wSelectFile.value) == 1:
            self.data = SfgRecord(
                self.folder + "/" + self.wSelectFile.value[0]
            )
        else:
            records = [SfgRecord(self.folder + "/" + fname)
                       for fname in self.wSelectFile.value]
            self.data = concatenate_list_of_SfgRecords(records)
        self._unobserve_figure()
        self._central_wl = None
        self._set_zero_time_subtraction(None)
        self._set_roi_trace_x_pixel()
        self._set_roi_frames()
        self._set_roi_spectra()
        self._set_roi_delays()
        self._set_pumped_index()
        # Deactivating the observers here prevents flickering
        # and unneeded calls of _update_figure. Thus we
        # call it manually after a recall of _init_observer
        self._conf_widget_with_data()
        self._init_figure_observers()
        self._update_figure()
        #print("keep figures unobserved: ", keep_figure_unobserved)

    def _set_roi_trace_x_pixel(self, new=None):
        self._rois_x_pixel_buffer[0] = slice(*self.wRangeSliderTracePixelX.value)
        self.data.rois_x_pixel_trace = self._rois_x_pixel_buffer

    def _set_roi_frames(self, new=None):
        self.data.roi_frames = slice(*self.wRangeSliderFrame.value)

    def _set_roi_spectra(self, new=None):
        self.data.roi_spectra = self.spec_slice

    def _set_roi_delays(self, new=None):
        self.data.roi_delays = self.wRangeSliderPPDelay.slice

    def _set_pumped_index(self, new=None):
        self.data.unpumped_index = self._unpumped_index_buffer
        self.data.pumped_index = self._pumped_index_buffer

    def x_spec_renew(self, new={}):
        """Renew calibration according to gui."""
        cw = self.wTextCentralWl.value
        vis_wl = self.wTextVisWl.value
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

    def _on_delay_median_clicked(self, new=None):
        if self.wCheckDelayMedian.value:
            self.wDropdownDelayMode.value = "Region"

    def _on_frame_median_clicked(self, new=None):
        if self.wCheckFrameMedian.value:
            self.wDropdownFrameMode.value = "Region"
            self.wSliderFrame.disabled = True
        else:
            self.wSliderFrame.disabled = False

    def _on_frame_mode_changed(self, new=None):
        if self.wDropdownFrameMode.value == "Index":
            self.wCheckFrameMedian.value = False

    def _on_delay_mode_changed(self, new=None):
        if self.wDropdownDelayMode.value == "Region":
            self.wSliderPPDelay.disabled = True
        else:
            self.wSliderPPDelay.disabled = False

    def _on_calib_changed(self, new=None):
        """Calibration changed."""
        self._toggle_vis_wl()
        self._toggle_central_wl()
        self.x_spec_renew()
        self.wCheckAutoscale.value = True

    def _on_pumped_index_changed(self, new=None):
        """Reset Bleach related properties."""
        self._pumped_index_buffer = self.wIntTextPumped.value
        self.data.pumped_index = self.wIntTextPumped.value

    def _on_unpumped_index_changed(self, new=None):
        self._unpumped_index_buffer = self.wIntTextUnpumped.value
        self.data.unpumped_index = self.wIntTextUnpumped.value

    def _set_zero_time_subtraction(self, new=None):
        self.data.zero_time_subtraction = \
            self.wCheckShowZeroTimeSubtraction.value

    def _set_zero_time_selec(self, new=None):
        self.data.zero_time_selec = self.wRangeZeroTime.slice

    def _on_baseline_offset_changed(self, new=None):
        self.data.baseline_offset = self.wTextBaselineOffset.value

    def _on_save_record(self, new=None):
        fname = self.folder + '/' + self.wTextSaveRecord.value
        self.data.save(fname)

    def _snap_x_roi(self, new=None):
        self.data.rois_x_pixel_trace.append(self.wRangeSliderTracePixelX.slice)
        # We want to be able to save the snaps throghout different data sets.
        self._rois_x_pixel_buffer = self.data.rois_x_pixel
        self._update_figure()

    @property
    def central_wl(self):
        """Central wl used for x axis calibration of the *Spectrum* axis."""
        if not self._central_wl:
            if self.data.metadata.get('central_wl') != 0:
                self._central_wl = self.data.metadata.get('central_wl')
        return self._central_wl

    @property
    def vis_wl(self):
        """The wavelength of the visible.

        The visible wavelength is used as upconversion number during the
        calculation of the wavenumber values of the x axis of the *Signal*
        plot."""
        if not self._vis_wl:
            return 0
        return self._vis_wl

    @property
    def folder(self):
        return os.path.abspath(self.wTextFolder.value)

    @property
    def fnames(self):
        return _filter_fnames(self.wTextFolder.value)

    @property
    def pp_delay_slice(self):
        """PP Delay index Slice"""
        return self.wRangeSliderPPDelay.slice

    @property
    def pp_delay_selected(self):
        if self.wDropdownDelayMode.value == "Index":
            return _slider_int_to_slice(self.wSliderPPDelay)
        return self.wRangeSliderPPDelay.slice

    @property
    def frame_selected(self):
        """Gui selected frame slice."""
        if self.wDropdownFrameMode.value == "Index":
            return _slider_int_to_slice(self.wSliderFrame)
        return self.wRangeSliderFrame.slice

    @property
    def spec_slice(self):
        """Specta slice/Y-Pixel slice."""
        sl = self.wRangeSliderPixelY.slice
        ret = slice(sl.start, sl.stop, self.wIntTextPixelYStep.value)
        return ret

    @property
    def x_pixel_slice(self):
        return self.wRangeSliderPixelX.slice

    @property
    def x_trace_pixel_slice(self):
        """X Pixel slice."""
        return self.wRangeSliderTracePixelX.slice

    @property
    def x_spec(self):
        """X data of the *Signal* plot. """
        if self.wDropdownCalib.value == 'pixel':
            x = self.data.pixel
        elif self.wDropdownCalib.value == 'wavelength':
            x = self.data.wavelength
        elif self.wDropdownCalib.value == 'wavenumber':
            x = self.data.wavenumber
        return x

    def select_spectra(self, y_property):
        """Use settings of the gui to subselect spectra data from SfgRecord."""
        kwgs = dict(
            y_property=y_property,
            x_property=self.wDropdownCalib.value,
            roi_delay=self.pp_delay_selected,
            roi_frames=self.frame_selected,
            roi_spectra=self.spec_slice,
            roi_x_pixel_spec=self.x_pixel_slice,
            frame_med=self.wCheckFrameMedian.value,
            delay_mean=self.wCheckDelayMedian.value,
            spectra_mean=self.wCheckSpectraMean.value,
            medfilt_pixel=self.wIntSliderSmooth.value,
        )
        return self.data.subselect(**kwgs)

    def select_trace(self, y_property):
        """Use settings of gui to susbelect data for trace."""
        kwgs = dict(
            y_property=y_property,
            x_property='pp_delays',
            roi_delay=self.wRangeSliderPPDelay.slice,
            roi_frames=self.frame_selected,
            roi_spectra=self.spec_slice,
            roi_x_pixel_spec=self.x_trace_pixel_slice,
            frame_med=self.wCheckFrameMedian.value,
            spectra_mean=self.wCheckSpectraMean.value,
            pixel_mean=True,
            medfilt_pixel=self.wIntSliderSmooth.value,
        )
        return self.data.subselect(**kwgs)

    @property
    def x_vlines(self):
        ret = [self.x_spec[self.x_trace_pixel_slice.start],
               self.x_spec[self.x_trace_pixel_slice.stop - 1]]
        return ret


class WidgetPlots():
    """Plotly Base plotting backend."""
    def __init__(self):
        import plotly.graph_objs as go
        # Plotly figure obj
        self.figure = go.Figure()
        # List of plotly data object to plot on the figure
        self.data = []
        # Plotly layout obj for the figure.
        self.layout = go.Layout()

    def _update_figure(self):
        pass

    def _init_figure(self):
        pass


class WidgetFigures():
    """Collect figure init and update functions within this class"""
    axes_grid = np.array([[]])  # a 2d array with the figure axes

    @property
    def fig(self):
        return self._fig

    @property
    def axes(self):
        return self._fig.axes

    def redraw_figure(self):
        """This forces matplotlib to update the figure canvas."""
        self._fig.canvas.draw()
        for ax in self.axes:
            ax.figure.canvas.draw()

    def _update_figure(self):
        # OVERWRITE THIS FUNCTION
        pass

    def _update_figure_callback(self, new):
        """A callback version of _update_figure for usage in observers."""
        self._update_figure()

    def init_single_figure(self):
        """Init the fiures and axes"""
        try:
            conds = (
                Counter(self.axes) != Counter(
                    self.axes_grid.flatten().tolist()
                ),
                len(self.axes) is 0
            )
            if not self._fig:
                self._fig, self.axes_grid = plt.subplots(
                    1, 1, figsize=self._figsize, squeeze=False
                )
            # This allows for redrawing the axis on an already existing figure.
            elif any(conds):
                self._fig.set_size_inches(self._figsize, forward=True)
                self.axes_grid = np.array([[self._fig.add_subplot(111)]])
        except TypeError:
            pass

    def init_two_figures(self):
        """Init the two axis figure."""
        try:
            conds = (
                Counter(self.axes) != Counter(
                    self.axes_grid.flatten().tolist()
                ),
                len(self.axes) is 0
            )
            if not self._fig:
                self._fig, self.axes_grid = plt.subplots(
                    1, 2, figsize=self._figsize, squeeze=False
                )
            # This allows for redrawing the axis on an already existing figure.
            elif any(conds):
                self._fig.set_size_inches(self._figsize, forward=True)
                self.axes_grid = np.array([[
                    self._fig.add_subplot(121),
                    self._fig.add_subplot(122)
                ]])
        except TypeError:
            pass

    def _plot_spec(self, xdata, ydata, ax, label_base=""):
        """Plot the basic 4d data types of the data record.


        xdata: The x_axis of the plot.
        ydata: 4d array.
        ax: matplotlib axis."""
        initial = True
        for delay_index in range(len(ydata)):
            delay = ydata[delay_index]
            for frame_index in range(len(delay)):
                frame = delay[frame_index]
                for spectrum_index in range(len(frame)):
                    spectrum = frame[spectrum_index]
                    if initial:
                        initial = False
                    else:
                        label_base = ''

                    label_str = self._append_identifier(label_base).format(
                        self.pp_delay_selected.start + delay_index,
                        self.pp_delay_selected.stop,
                        self.frame_selected.start + frame_index,
                        self.frame_selected.stop,
                        self.spec_slice.start + spectrum_index,
                        self.spec_slice.stop
                    )
                    ax.plot(xdata, spectrum, label=label_str)

    def _plot_traces(self, xdata, ydata, ax, label_base=''):
        initial = True
        y = ydata.T
        for pixel in y:
            for spec in pixel:
                for frame in spec:
                    #label_str = label_base + '{:.0f}-{:.0f}'.format(
                    #    x_region[0],
                    #    x_region[-1]
                    #)
                    ax.plot(xdata, frame.T, '-o',)

   #     for roi_index in range(len(data)):
   #         # data is of shape [roi, pp_delay, spectra]
   #         roi = data[roi_index]
   #         roi_slice = self.data.rois_x_pixel_trace[roi_index]
   #         if initial:
   #             initial = False
   #         else:
   #             label_base = ""
   #         x_region = np.sort(self.x_spec[roi_slice])
   #         label_str = label_base + '{:.0f}-{:.0f}'.format(
   #             x_region[0],
   #             x_region[-1]
   #         )
   #         ax.plot(self.x_trace, roi, "-o", label=label_str)

    def _plot_rawData(self, ax):
        if not self.wCheckShowRawData.value:
            return
        self._plot_spec(*self.select_spectra('rawData'), ax, 'RawData\n')

    def _plot_basesubed(self, ax):
        if not self.wCheckShowBasesubed.value:
            return
        self._plot_spec(*self.select_spectra('basesubed'), ax, 'Basesubed\n')

    def _plot_normalized(self, ax):
        if not self.wCheckShowNormalized.value:
            return
        self._plot_spec(*self.select_spectra('normalized'), ax, 'Normalized\n')

    def _plot_base(self, ax):
        if not self.wCheckShowBase.value:
            return
        self._plot_spec(*self.select_spectra('base'), ax, 'Base\n')

    def _plot_norm(self, ax):
        if not self.wCheckShowNorm.value:
            return
        self._plot_spec(*self.select_spectra('norm'), ax, 'Norm\n')

    def _plot_bleach_abs(self, ax):
        if not self.wCheckShowBleachAbs.value:
            return
        self._plot_spec(*self.select_spectra('bleach_abs'), ax, "BleachAbs\n")

    def _plot_bleach_abs_norm(self, ax):
        if not self.wCheckShowBleachAbsNorm.value:
            return
        self._plot_spec(*self.select_spectra('bleach_abs_norm'), ax, "BleachAbsNorm\n")

    def _plot_bleach_rel(self, ax):
        if not self.wCheckShowBleachRel.value:
            return
        self._plot_spec(*self.select_spectra("bleach_rel"), ax, "BleachRel\n")

    def _plot_bleach_rel_abs(self, ax):
        if not self.wCheckShowBleachRelNorm.value:
            return
        self._plot_spec(*self.select_spectra('bleach_rel_norm'), ax, "BleachRelNorm\n")

    def _plot_traces_bleach_abs(self, ax):
        if not self.wCheckShowTracesBleachAbs.value:
            return
        self._plot_traces(*self.select_trace('bleach_abs'), ax, 'Bleach Abs\n')

    def _plot_traces_bleach_abs_norm(self, ax):
        if not self.wCheckShowTracesBleachAbsNorm.value:
            return
        self._plot_traces(*self.select_trace('bleach_abs_norm'), ax,
                          'Bleach Abs Norm\n')

    def _plot_traces_bleach_rel(self, ax):
        if not self.wCheckShowTracesBleachRel.value:
            return
        self._plot_traces(*self.select_trace('bleach_rel'), ax, 'Bleach Rel\n')

    def _plot_traces_bleach_rel_norm(self, ax):
        if not self.wCheckShowTracesBleachRelNorm.value:
            return
        self._plot_traces(*self.select_trace('bleach_rel_norm'), ax,
                          'Bleach Rel Norm\n')

    def _plot_traces_rawData(self, ax):
        if not self.wCheckShowTracesRawData.value:
            return
        self._plot_traces(*self.select_trace('rawData'), ax, 'Bleach Raw\n')

    def _plot_traces_basesubed(self, ax):
        if not self.wCheckShowTracesBasesubed.value:
            return
        self._plot_traces(*self.select_trace('basesubed'), ax,
                          'Bleach Basesubed\n')

    def _plot_traces_normalized(self, ax):
        if not self.wCheckShowTracesNormalized.value:
            return
        self._plot_traces(*self.select_trace('normalized'), ax, 'Bleach Norm\n')

    def plot_spec(self, ax):
        self._plot_rawData(ax)
        self._plot_basesubed(ax)
        self._plot_normalized(ax)
        self._plot_base(ax)
        self._plot_norm(ax)
        self._plot_bleach_abs(ax)
        self._plot_bleach_abs_norm(ax)
        self._plot_bleach_rel(ax)
        self._plot_bleach_rel_abs(ax)
        ax.set_title(self._x_spec_title)
        ax.set_xlabel(self.x_spec_label)

    def plot_traces(self, ax):
        self._plot_traces_bleach_abs(ax)
        self._plot_traces_bleach_abs_norm(ax)
        self._plot_traces_bleach_rel(ax)
        self._plot_traces_bleach_rel_norm(ax)
        self._plot_traces_rawData(ax)
        self._plot_traces_basesubed(ax)
        self._plot_traces_normalized(ax)
        ax.set_xlabel('pp delay / fs')
        ax.set_title('Trace')
        ax.legend()

    def _on_ax0_lim_changed(self, new=None):
        """Callback for the *Signal* axis."""
        # Called when the xlim of the `Signal` plot is changed
        self._autoscale_buffer = _lims2buffer(self.axes[0])

    def _on_ax1_lim_changed(self, new=None):
        # Called when the ylim of the `Signal` plot is changed
        self._autoscale_buffer_2 = _lims2buffer(self.axes[1])

    @property
    def x_spec_label(self):
        """x axis label of the spec plot"""
        if self.wDropdownCalib.value == 'wavenumber':
            ret = r"Wavenumber/cm$^{-1}$"
        elif self.wDropdownCalib.value == 'wavelength':
            ret = "Wavelength/nm"
        else:
            ret = "Pixel"
        return ret

    @property
    def _x_spec_title(self):
        """Title of the spec plot."""
        if self.wDropdownDelayMode.value == 'Index':
            return "Delay {} fs".format(
                self.data.pp_delays[self.pp_delay_selected.start]
            )
        else:
            return "Delay {} - {} fs".format(
                self.x_trace[0], self.x_trace[-1]
            )

    def _append_identifier(self, label_base):
        """Append identifier to label string for plots."""
        if self.wCheckDelayMedian.value:
            label_base += 'D[{0}:{1}]_'
        else:
            label_base += 'D[{0}]_'
        if self.wCheckFrameMedian.value:
            label_base += 'F[{2}:{3}]_'
        else:
            label_base += 'F[{2}]_'
        if self.wCheckSpectraMean.value:
            label_base += 'S[{4}:{5}]'
        else:
            label_base += 'S[{4}]'
        return label_base


class BaselineTab(WidgetBase, WidgetFigures):
    def __init__(self, figsize=(8, 6), **kwargs):
        super().__init__(figsize=figsize, **kwargs)

    def _init_widget(self):
        """Init the widgets that are to be shown."""
        import ipywidgets as wi
        super()._init_widget()
        self.wRangeSliderTracePixelX.layout.visibility = 'hidden'
        self.wCheckAutoscaleTrace.layout.visibility = 'hidden'
        self.wCheckShowBase.value = False
        self.children = wi.VBox([
            self._data_box,
            self._signal_box,
            self._calib_box,
            self._save_record_box
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
        ax.legend(framealpha=0.5)
        ax.set_xlabel(self.wDropdownCalib.value)
        ax.set_title('Baseline')
        ax.callbacks.connect('xlim_changed', self._on_ax0_lim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ax0_lim_changed)
        if self.wCheckAutoscale.value:
            self._autoscale_buffer_2 = _lims2buffer(ax)
        else:
            _buffer2lims(ax, self._autoscale_buffer_2)
        self.redraw_figure()

    @property
    def to_base(self):
        """Y data to be send on Set Baseline button press."""
        return self.select_spectra('rawData')[1]


class IRTab(WidgetBase, WidgetFigures):
    """Widget to visualize IRTab type data.

    IRTab Type data has a SfrRecord and a BaselineTab """

    def __init__(self, figsize=(8, 6), **kwargs):
        super().__init__(figsize=figsize, **kwargs)

    def _init_widget(self):
        """Init the widgets that are to be shown."""
        import ipywidgets as wi
        super()._init_widget()
        # This allows the data to be used for normalization from start on
        self.data.data += 1
        self.wRangeSliderTracePixelX.layout.visibility = 'hidden'
        self.wCheckAutoscaleTrace.layout.visibility = 'hidden'
        self.wCheckShowBase.value = False
        show_box = wi.HBox([
            self.wCheckShowRawData,
            self.wCheckShowBasesubed,
            self.wCheckShowBase,
        ])
        self.children = wi.VBox([
            self._data_box,
            self._signal_box,
            self.wTextBaselineOffset,
            show_box,
            self._calib_box,
            self._save_record_box
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

    def _init_observer(self):
        super()._init_observer()

    @property
    def to_norm(self):
        """The property that gets exported to the Record tab if one clickes.
        Send IR."""
        return self.select_spectra('basesubed')[1]


class RecordTab(WidgetBase, WidgetFigures):
    def __init__(self, central_wl=None, vis_wl=810, figsize=(10, 4), **kwargs):
        """Plotting gui based on the SfgRecord class as a data backend.

        Parameters
        ----------
        data : Optional, SfgRecord obj.
            Default dataset to start with. If None, an empty one is created.
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
        test = RecordTab()
        test()
        # Type the Folder you want to investigate in the folder Text box and
        # press RETURN.
        # A list of selectable files will appear on the right side.
        """
        super().__init__(central_wl=central_wl, vis_wl=vis_wl, figsize=figsize,
                         **kwargs)
        self._ax_xlim = None
        self._ax_ylim = None

    def _init_figure(self):
        """Init the two axis figure."""
        self.init_two_figures()
        # TODO Axes is too small on summed

    def _init_widget(self):
        """Init all widgets that are to be drawn."""
        import ipywidgets as wi
        super()._init_widget()
        # self.children is the widget we are rendering up on call.
        show_box = wi.VBox([
            wi.HBox([
                self.wSnapXRoi,
                self.wTextBaselineOffset,
            ]),
            wi.HBox([
                self.wCheckShowRawData,
                self.wCheckShowBasesubed,
                self.wCheckShowBase,
                self.wCheckShowNorm,
                self.wCheckShowNormalized,
            ]),
            wi.VBox([
                wi.Label("Bleach:"),
                wi.HBox([
                   self.wCheckShowBleachAbs,
                   self.wCheckShowBleachAbsNorm,
                   self.wCheckShowBleachRel,
                   self.wCheckShowBleachRelNorm,
                ]),
            ]),
            wi.VBox([
                wi.Label("Traces:"),
                wi.HBox([
                    self.wCheckShowTracesRawData,
                    self.wCheckShowTracesBasesubed,
                    self.wCheckShowTracesNormalized,
                    self.wCheckShowTracesBleachAbs,
                    self.wCheckShowTracesBleachAbsNorm,
                    self.wCheckShowTracesBleachRel,
                    self.wCheckShowTracesBleachRelNorm,
                ]),
            ]),
        ])
        bleach_box = wi.HBox([
            self.wIntTextPumped,
            self.wIntTextUnpumped,
            self.wCheckShowZeroTimeSubtraction,
            self.wRangeZeroTime,
        ])
        self.children = wi.VBox([
            self._data_box,
            self._signal_box,
            show_box,
            bleach_box,
            self._calib_box,
            self._save_record_box
        ])

    def _update_figure(self):
        """Update the figure of the gui."""
        self._init_figure()

        fontsize = 8
        ax = self.axes[0]
        ax.clear()
        self.plot_spec(ax)
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax.callbacks.connect('xlim_changed', self._on_ax0_lim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ax0_lim_changed)
        if self.wCheckAutoscale.value:
            self._autoscale_buffer = _lims2buffer(ax)
        else:
            _buffer2lims(ax, self._autoscale_buffer)
        ax.vlines(self.x_vlines, *ax.get_ylim(),
                  linestyle="dashed")
        ax.legend(framealpha=0.5)

        ax = self.axes[1]
        ax.clear()
        self.plot_traces(ax)
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3g'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax.yaxis.tick_right()
        ax.callbacks.connect('xlim_changed', self._on_ax1_lim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ax1_lim_changed)
        if self.wCheckAutoscaleTrace.value:
            self._autoscale_buffer_2 = _lims2buffer(ax)
        else:
            _buffer2lims(ax, self._autoscale_buffer_2)

        self.redraw_figure()


# This is broken
class ImgView(WidgetBase):
    """A Class to view full spe images."""
    def __init__(self, *args, figsize=(8, 6), **kwargs):
        super().__init__(*args, figsize=figsize, **kwargs)
        self.axes_grid = np.array([[]])

    def _init_figure(self):
        if not self._fig:
            self._fig = plt.figure(self._figsize)
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3],
                                   height_ratios=[3, 1])
            ax = self._fig.add_subplot(gs[0, 1])
            self._fig.add_subplot(gs[0, 0], sharey=ax)
            self._fig.add_subplot(gs[1, 1], sharex=ax)
        elif self._fig and len(self.axes) is not 3:
            self._fig.set_size_inches(self._figsize, forward=True)
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3],
                                   height_ratios=[3, 1])
            ax = self._fig.add_subplot(gs[0, 1])
            self._fig.add_subplot(gs[0, 0], sharey=ax)
            self._fig.add_subplot(gs[1, 1], sharex=ax)

    def _update_figure(self):
        self._init_figure()
        view_data = self.data.data[
            self.pp_delay_index, self.frame_index
        ]
        ax = self.axes[0]
        plt.sca(ax)
        ax.clear()
        img = ax.imshow(
            view_data,
            interpolation=self.w_interpolate.value,
            origin="lower",
            aspect="auto"
        )
        plt.colorbar(img)

        axl = self.axes[1]
        axl.clear()
        y_slice = self.wRangeSliderPixelY.slice
        view_data2 = self.data.data[
            self.pp_delay_selected.start, self.wRangeSliderFrame.value[0], y_slice
        ].sum(Y_PIXEL_INDEX)
        axl.plot(view_data2)

    def _init_widget(self):
        import ipywidgets as wi
        super()._init_widget()
        self.wIntSliderSmooth.visible = False
        self.wIntSliderSmooth.disabled = True
        self.wRangeSliderPPDelay.visible = False
        self.wRangeSliderPPDelay.disabled = True
        self.w_interpolate = wi.Dropdown(
            description="Interpolation",
            options=('none', 'nearest', 'bilinear', 'bicubic',
                     'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
                     'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel',
                     'mitchell', 'sinc', 'lanczos'),
            value="nearest",
        )
        self.children = wi.VBox([
            self.wVBoxSignal,
            wi.HBox(
                [self.wDropdownCalib, self.wTextCentralWl, self.wTextVisWl]
            ),
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


class PumpProbe():
    """Tabed dashboard.

    The first page shows two axis.
    On the first axes one sees the raw signal. And possibly
    a baseline. Each y-pixel of the ccd camera gets projected into a single.
    spectra line on this first axes. With the *Show Baseline* Button one can
    toggle the visibility of the Baseline. Autoscale prevents the axes from
    re-scaling up on data change. Numorus sliders allow for inspection of the
    data.
    The second axes shows the Trace of each spectrum vs pump-probe time delay.
    This is only use full if you do pump-probe experiment. Otherwise this axis
    will only show to you the a single point with the value of the sum(area) of
    the spectrum from axes one.


    The second page shows A single Spectrum and possibly a baseline.

    The third page shows, after usage of the normalize button the quotient
    of the first and the second page spectrum. This allows for IRTab
    Normalization."""

    def __init__(self):
        import ipywidgets as wi
        self.tabed_widgets = (
            RecordTab(),
            IRTab(),
            BaselineTab(),
        )
        self.tab_record = self.tabed_widgets[0]
        self.tab_ir = self.tabed_widgets[1]
        self.tab_record_baseline = self.tabed_widgets[2]

        children = []
        self.wi_fig = plt.figure()
        # Names given explicitly to preserver order of tabs.
        for tabed_widget in self.tabed_widgets:
            tabed_widget._conf_widget_with_data()
            children.append(tabed_widget.children)
            tabed_widget._fig = self.wi_fig

        self.w_tabs = wi.Tab(children=children)
        names = ("Pump-Probe", "IR", "Baseline")
        for i in range(len(names)):
            self.w_tabs.set_title(i, names[i])
        self.children = self.w_tabs
        self.wButtonSetBaseline = wi.Button(description='Set Baseline')
        self.wButtonSetIrBaseline = wi.Button(description='Set Ir Baseline')
        self.wButtonNormalize = wi.Button(description='Set Normalize')
        self.wButtonSaveGui = wi.Button(description='Save Gui')
        self.wButtonLoadGui = wi.Button(description='Load Gui')
        self.children = wi.VBox([
            self.w_tabs,
            wi.HBox([
                self.wButtonSetBaseline,
                self.wButtonSetIrBaseline,
                self.wButtonNormalize,
                self.wButtonSaveGui,
                self.wButtonLoadGui,
            ])
        ])

    def __call__(self):
        from IPython.display import display
        for tabed_widget in self.tabed_widgets:
            tabed_widget._init_observer()
        self._init_observer()
        # Render record tab as default.
        self.tab_record._update_figure()
        display(self.children)

    def _init_observer(self):
        """Initialize widgets of the GUI.
        Observers within this function interact between tabs, of independent
        of tabs.
        """
        if debug:
            print("Dasboards._init_observer called")

        def test_widgets(tab):
            """List of widgets of a tab that change the data such that,
            test must be run before Ir or Baseline can be set."""
            return (
                tab.wSliderPPDelay,
                tab.wRangeSliderPPDelay,
                tab.wCheckDelayMedian,
                tab.wDropdownDelayMode,
                tab.wSliderFrame,
                tab.wRangeSliderFrame,
                tab.wCheckFrameMedian,
                tab.wDropdownFrameMode,
                tab.wIntSliderSmooth,
                tab.wRangeSliderPixelY,
                tab.wIntTextPixelYStep,
                tab.wSelectFile,
            )

        self.w_tabs.observe(self._on_tab_changed, 'selected_index')
        self.wButtonSetBaseline.on_click(self._on_setBaseline_clicked)
        self.wButtonSetIrBaseline.on_click(self._on_setIRBaseline_clicked)
        self.wButtonNormalize.on_click(self._on_set_normalize)
        for widget in test_widgets(self.tab_ir):
            widget.observe(
                self._test_normalizability,
                "value"
            )
        for widget in test_widgets(self.tab_record_baseline):
            widget.observe(self._test_Record_baseline, "value")
            widget.observe(self._test_IR_Baseline, "value")

        self.tab_record_baseline.wSelectFile.observe(
            self._test_Record_baseline,
            "value"
        )
        self.tab_record_baseline.wSelectFile.observe(
            self._test_normalizability,
            "value"
        )
        self.wButtonSaveGui.on_click(self._on_save_gui_clicked)
        self.wButtonLoadGui.on_click(self._on_load_gui_clicked)

    def _on_tab_changed(self, new):
        if debug:
            print("Dashboard._on_tab_changed called")
        axes = self.wi_fig.axes
        for ax in axes:
            self.wi_fig.delaxes(ax)
        page = self.w_tabs.selected_index
        widget = self.tabed_widgets[page]
        widget._update_figure()

    def _on_setBaseline_clicked(self, new):
        """Called when set baseline is clicked."""
        if not self._test_baseline_on_tab(
                self.tab_record, self.wButtonSetBaseline
        ):
            return
        self.tab_record.data.base = self.tab_record_baseline.to_base
        self.wButtonSetBaseline.style.button_color = "green"
        self.tabed_widgets[self.w_tabs.selected_index]._update_figure()

    def _on_setIRBaseline_clicked(self, new):
        """Called when set ir baseline is clicked."""
        if not self._test_baseline_on_tab(
                self.tab_ir, self.wButtonSetIrBaseline
        ):
            return
        self.tab_ir.data.base = self.tab_record_baseline.to_base
        self.wButtonSetIrBaseline.style.button_color = "green"
        self.tabed_widgets[self.w_tabs.selected_index]._update_figure()

    def _on_set_normalize(self, new):
        if debug:
            print("Normalize._on_set_normalize called.")
        if not self._test_normalizability():
            return
        self.tab_record.data.norm = self.tab_ir.to_norm
        # Update current plot
        self.wButtonNormalize.style.button_color = "green"
        self.tabed_widgets[self.w_tabs.selected_index]._update_figure()

    def _on_save_gui_clicked(self, new):
        """Save gui status to a json text file.

        Each tab of the dashboard gets a separate list entry. Each widget value
        is saved as an dictionary of widget names and values."""
        save_file = self.tab_record.folder + '/.last_state.json'
        with open(save_file, 'w') as outfile:
            save_list = []
            for i in range(len(self.tabed_widgets)):
                w = self.tabed_widgets[i]
                save_dict = {}
                for name, saveable_widget in w._save_widgets.items():
                    save_dict[name] = saveable_widget.value
                save_list.append(save_dict)
            dump(save_list, outfile, indent=4,
                 separators=(',', ': '), sort_keys=True)

    def _on_load_gui_clicked(self, new):

        def _pop_and_set(name):
            value = saved_values.pop(name)
            w._save_widgets[name].value = value

        def _read_and_set(name):
            value = saved_values.get(name)
            widget = w._save_widgets.get(name)
            if isinstance(value, type(None)):
                return
            if isinstance(widget, type(None)):
                return

            widget.value = value

        try:
            infile = open(self.tab_record.folder + '/.last_state.json', 'r')
        except FileNotFoundError:
            pass
        else:
            with infile:
                imp = load(infile)
                # Loop over tabs
                for i in range(len(self.tabed_widgets)):
                    saved_values = imp[i]
                    w = self.tabed_widgets[i]
                    # read folder file and baseline as the first
                    _pop_and_set('folder')
                    w._on_folder_submit(None)
                    _pop_and_set('file')
                    w._load_data()
                    w._unobserve_figure()
                    for name in saved_values.keys():
                        try:
                            _read_and_set(name)
                        except TraitError:
                            msg = "Can't load {} with value {}".format(
                                name,
                                saved_values[name]
                            )
                            print(msg)
                            break
                    w._init_figure_observers()
            self._on_tab_changed(None)

    def _test_normalizability(self, new=None):
        """Test if the data of w1 can be used to normalize the data of w0."""
        try:
            norm = np.ones_like(self.tab_record.data.rawData) *\
                   self.tab_ir.to_norm
            self.wButtonNormalize.style.button_color = 'orange'
            if np.all(self.tab_record.data.norm == norm):
                self.wButtonNormalize.style.button_color = 'green'
        except ValueError:
            self.wButtonNormalize.style.button_color = 'red'
            return False
        return True

    def _test_baseline_on_tab(self, tab, button):
        """Test if baseline data of tab can be setted.

        tab: tab to set the baselinedata of
        button: button that was clicked and that sould be colored accordingly.
        """
        try:
            base = np.ones_like(tab.data.rawData) *\
                self.tab_record_baseline.to_base
            button.style.button_color = 'orange'
            # Must use _base here because .base has offset correction.
            if isinstance(tab.data._base, type(None)):
                return True
            if np.all(tab.data._base == base):
                button.style.button_color = 'green'
        except ValueError:
            button.style.button_color = 'red'
            return False
        return True

    def _test_IR_Baseline(self, new=None):
        return self._test_baseline_on_tab(
            self.tab_ir,
            self.wButtonSetIrBaseline
        )

    def _test_Record_baseline(self, new=None):
        return self._test_baseline_on_tab(
            self.tab_record,
            self.wButtonSetBaseline
        )

# #### Helper function
def _filter_fnames(folder_path):
    """Return list of known files in a folder."""

    fnames = np.sort(glob(os.path.normcase(folder_path + '/*')))
    # Only .dat, .spe and .npz are known
    mask = [
        any(conds) for conds in zip(
            [".dat" in s for s in fnames],
            [".spe" in s for s in fnames],
            [".npz" in s for s in fnames],
        )
    ]
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


def _slider_int_to_slice(slider):
    return slice(slider.value, slider.value+1)


def to_slice(attribute):
    # This can be used as a decorator, to get slices from Rangedwidgets
    # I'm currently not using it, beacuse I think its more complicated,
    # then explicitly calling the rangeSlider_to_slice function on the
    # Sliders.
    def _to_slice(f):
        def wrapper(self, *args):
            widget = getattr(self, attribute)
            return slice(*widget.value)
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


def _set_rangeSlider_num_to_label(lines, sliceObj, label_base=""):
    """Use a rangeSlider, to add rangeSlider values to label_base

    lines: The lines to set the label of.
    y_slice: The rangeSlider to extract values from
    label_base: base string of the label that the number is appended to."""
    j = 0
    for i in range(*sliceObj.indices(sliceObj.stop)):
        label = label_base + str(i)
        line = lines[j]
        line.set_label(label)
        j += 1


class IntRangeSliderGap(IntRangeSlider):
    """A Ranged slider with enforced gap."""
    @validate('value')
    def enforce_gap(self, proposal):
        gap = 1
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

    @property
    def slice(self):
        return slice(*self.value)

#### End of helper functions
