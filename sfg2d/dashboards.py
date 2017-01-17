"""A Module for Dashboards.

A Dasboard is a high level GUI class that combines many widgets.
Such a dashboard is then used during experiment to analyze data.

Dasboards typically take classes from widgets.py and combine them.
A dashboard does not inherit from any widget class it self. If
 you find your self inheriting from a widget subclass, then chances
are, you are writing a widget."""

from . import widgets as wi

debug = 0

class Dashboard():
    def __init__(self, *args, **kwargs):
        self.widgets = args
        self.fig = None
        self.ax = None

    def __call__(self):
        pass

class Normalize(Dashboard):
    """A 3 Pages Tabed dashboard.

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
        import ipywidgets as iwi
        super().__init__(*args, **kwargs)
        children = []
        self.wi_fig = wi.plt.figure()
        for widget in args:
            widget._configure_widgets()
            children.append(widget.children)
            widget._fig = self.wi_fig

        self.w_tabs = iwi.Tab(children=children)
        self.children = self.w_tabs
        self.w_normalize = iwi.Button(description='Normalize')
        self.children = iwi.VBox([self.w_tabs, self.w_normalize])

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
        w0.w_y_pixel_range.observe(self._is_normalizable_callback, "value")
        w1.w_y_pixel_range.observe(self._is_normalizable_callback, "value")

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
        ir = wi.np.ones_like(spec) * w1.y.T

        w2.data = w0.data.copy()
        w2.data.data = spec/ir
        w2._unobserve()
        if self.w_tabs.selected_index is 2:
            w2._update_figure()
        w2._configure_widgets()
        w2._init_observer()

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
