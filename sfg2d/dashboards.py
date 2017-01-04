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

    def __call__(self):
        from IPython.display import display
        #self.fig = wi.plt.figure()
        for widget in self.widgets:
            widget._init_observer()
            #widget._fig = self.fig
        self._init_observer()
        display(self.w_tabs)

    def _init_observer(self):
        if debug:
            print("Dasboards._init_observer called")
        self.w_tabs.observe(self._on_tab_changed, 'selected_index')

    def _on_tab_changed(self, new):
        if debug:
            print("Dashboard._on_tab_changed called")
        axes = self.fig.axes
        for ax in axes:
             self.fig.delaxes(ax)
        page = self.w_tabs.selected_index
        widget = self.widgets[page]
        widget._update_figure()
