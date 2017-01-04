"""A Module for Dashboards.

A Dasboard is a high level GUI class that combines many widgets.
Such a dashboard is then used during experiment to analyze data.

Dasboards typically take classes from widgets.py and combine them.
A dashboard does not inherit from any widget class it self. If
 you find your self inheriting from a widget subclass, then chances
are, you are writing a widget."""

from . import widgets as wi

debug = 1

class Dashboard():
    def __init__(self, *args, **kwargs):
        self.widgets = args
        self.fig = None
        self.ax = None

    def __call__(self):
        pass

class Tabulated(Dashboard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import ipywidgets as iwi

        children = []
        self.fig = wi.plt.figure()
        for widget in args:
            widget._configure_widgets()
            children.append(widget.children)
            widget._fig = self.fig

        self.w_tabs = iwi.Tab(children=children)
        self.w_normalize = iwi.ToggleButton(description='Normalize')
        self.children = iwi.VBox([self.w_tabs, self.w_normalize])

    def __call__(self):
        from IPython.display import display
        #self.fig = wi.plt.figure()
        for widget in self.widgets:
            widget._init_observer()
            #widget._fig = self.fig
        self._init_observer()
        display(self.children)

    def _init_observer(self):
        if debug:
            print("Dasboards._init_observer called")
        self.w_tabs.observe(self._on_tab_changed, 'selected_index')
        # Normalization must be registered to all widgets on the second page
        # to avoid divition by 0 errors.
        #for widget in self.widgets[1].children.children:
        #    widget.observe(self._toggle_normalization, 'value')
        self.w_normalize.observe(self._on_normalize, 'value')

    @property
    def y_normalized(self):
        spec = self.widgets[0].y
        ir = self.widgets[1].y
        return spec/ir

    def _on_tab_changed(self, new):
        if debug:
            print("Dashboard._on_tab_changed called")
        axes = self.fig.axes
        for ax in axes:
             self.fig.delaxes(ax)
        page = self.w_tabs.selected_index
        widget = self.widgets[page]
        widget._update_figure()

    def _on_normalize(self, new):
        """Set gui in such a state, that it can savely normalize."""
        pass

    def _check_ir_and_spec_ranges(self):
        w0, w1,  = self.widgets
        if w0.y.shape[0] != w1.y.shape[0]:
            self.w_normalize.disable = True
            return False
        if w0.y.shape[1] != w1.y.shape[1]:
            self.w_normalize.disabled = True
            return False
        self.w_normalize.disabled = False
        return True
