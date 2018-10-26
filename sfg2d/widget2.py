import os
from IPython.display import display, update_display
import ipywidgets as widgets
import sfg2d.raw_reader as raw_reader
from traitlets import validate
import logging
import numpy as np
import dpath
import yaml


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# Dict of loaded records.
records = {}

class MyWidgetBase():
    def __call__(self):
        display(self.widget)

class IntRangeSliderGap(widgets.IntRangeSlider):
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

class ValueChecker(MyWidgetBase):
    def __init__(self, value=None, description=None):
        """Widget to have checkbox for regions"""
        self._value = None
        self.widgets = []
        self.description = widgets.Label(
            description,
            layout={'width': '150px'}
        )
        self.widget = widgets.HBox([
            self.description
        ])
        if value:
            self.value = value

    @property
    def value(self):
        if not self._value:
            return None
        ret = np.array(self._value)[([wg.value for wg in self.widgets])]
        return ret.tolist()

    @value.setter
    def value(self, value):
        if isinstance(value, type(None)):
            return
        if isinstance(value, int):
            value = list(range(value))
        if len(value) >= 6:
            logger.warning(
                'Setting to big value makes label: "{}" disappear'.format(
                    self.description.value
                )
            )
        self._value = value
        self.widgets = []
        for elm in value:
            self.widgets.append(widgets.Checkbox(
                value=True,
                description='{}'.format(elm),
            ))
        self.widget.children = (self.description, *self.widgets,)

class FilePaths(MyWidgetBase):
    """A widget to deal with string or list of strings input at the same time"""
    def __init__(self, value=None, description=None):
        if isinstance(value, str):
            value = [value]

        label = widgets.Label(
            description,
            layout={'width': '150px'},
        )
        self.add = widgets.Button(
            description='+',
            layout={'width': '50px'},
        )
        self.sub = widgets.Button(
            description='-',
            layout={'width': '50px'},
        )

        list_of_paths = []
        if value:
            for elm in value:
                list_of_paths.append(
                    widgets.Text(
                        value=elm,
                    )
                )
        self.list_of_paths_box = widgets.VBox(list_of_paths)

        self.widget = widgets.VBox([
            widgets.HBox([label, self.add, self.sub]),
            self.list_of_paths_box
        ])
        self.init_observer()

    def on_add(self, change):
        w = widgets.Text()
        self.list_of_paths_box.children += (w,)

    def on_sub(self, change):
        self.list_of_paths_box.children = (*list(self.list_of_paths_box.children)[:-1], )

    @property
    def value(self):
        ret = []
        for widget in self.list_of_paths_box.children:
            v = widget.value
            if v == '':
                continue
            ret.append(v)
        return ret

    def init_observer(self):
        self.add.on_click(self.on_add)
        self.sub.on_click(self.on_sub)

class Kwargs(MyWidgetBase):
    def __init__(self, value=None, *args, **kwargs):
        """value: dict of kwargs."""
        self.text = widgets.Textarea(*args, **kwargs)
        self.value = value
        self.widget = self.text
        self.text.observe(self._on_widget_changed, names='value')

    def _update_text(self):
        """Update text reprs of value on widget."""
        self.text.value = yaml.dump(self._value, default_flow_style=False)

    @property
    def value(self):
        #value = yaml.load(self.text.value)
        return self._value

    @value.setter
    def value(self, value):
        if not value:
            value = {}
        self._value = value
        self._update_text()

    def set(self, key, value):
        """Function to set a key in value and update widget at the same time."""
        self._value[key] = value
        self._update_text()

    def pop(self, key):
        ret = self._value.pop(key)
        self._update_text()
        return ret

    def _on_widget_changed(self, change):
        self._value = yaml.load(self.text.value)


class Record(MyWidgetBase):
    """
    Build a gui interface for the `sfg2d.raw_reader.import_record`
    function. config is the config dict this function takes.
    """
    def __init__(
            self, config=None
    ):
        # dict of all config values for this widget.
        if not config:
            config = {}
        self.config = config
        # Current record object
        kwargs_record = config.get('kwargs_record', {})
        roi_spectra = kwargs_record.get('roi_spectra')
        roi_x_pixel_spec = (
            kwargs_record.get('roi_x_pixel_spec', slice(0, 1600)).start,
            kwargs_record.get('roi_x_pixel_spec', slice(0, 1600)).stop
            )
        vis_wl = kwargs_record.get('vis_wl', 812)

        # Widget to enter a folder path as string
        self.name = widgets.Text(
            value=config.get('name'),
            description='Name',
        )
        self.load = widgets.Button(
            description='Load'
        )
        self.fpath = FilePaths(
            value=config.get('fpath'),
            description='Path'
        )
        self.base = widgets.Text(
            value=config.get('base'),
            description='Baseline',
        )
        self.norm = widgets.Text(
            value=config.get('norm'),
            description='Normalization',
        )

        # Settet via kwargs
        self.roi_spectra = ValueChecker(
            value=roi_spectra,
            description='Roi Spectra'
        )
        self.roi_x_pixel_spec = IntRangeSliderGap(
            value=roi_x_pixel_spec,
            description='Roi X Spectra',
            min=0, max = 1600,
        )
        self.vis_wl = widgets.FloatText(
            value=vis_wl,
            description='Vis Wl',
            disabled=True,
        )
        self.vis_wl_disabler = widgets.Checkbox(
            description='Automatic',
            value=True,
        )

        self.pumped_index = widgets.BoundedIntText(
            description='Pumped',
            disabled=True,
        )
        self.unpumped_index = widgets.BoundedIntText(
            value=1,
            description='Unpumped',
            disabled=True,
        )
        self.pump_unpumped_disabler = widgets.Checkbox(
            description='Disable',
            value=True
        )
        #self.kwargs_record = widgets.Textarea(
        #    value=str(kwargs_record),
        #    description='kwargs'
        #)
        self.kwargs_record = Kwargs(
            value=kwargs_record,
            description='kwargs',
            layout={'width': '600px', 'height': '200px'},
        )
        # list of all real widgets
        self._widgets = [
            self.load,
            self.name,
            self.fpath.widget,
            self.base,
            self.norm,
            self.roi_spectra.widget,
            self.roi_x_pixel_spec,
            self.vis_wl,
            self.pumped_index,
            self.unpumped_index,
            self.kwargs_record.widget,
            self.vis_wl_disabler,
            self.pump_unpumped_disabler,
        ]
        # Kwargs Accordion
        ac_opt = widgets.Accordion([widgets.VBox([
            self.roi_spectra.widget,
            self.roi_x_pixel_spec,
            widgets.HBox([self.vis_wl, self.vis_wl_disabler]),
            widgets.HBox([self.pumped_index, self.unpumped_index, self.pump_unpumped_disabler]),
            self.kwargs_record.widget,
        ])])
        ac_opt.set_title(0, "Kwargs")

        # Options Accordion
        ac = widgets.Accordion(
            children=[
                widgets.VBox([
                    widgets.HBox([
                       self.fpath.widget,
                       self.load
                    ]),
                    self.base,
                    self.norm,
                    ac_opt,
                ])
            ],
        )
        ac.set_title(0, "Options")

        # The metawidget constructed from this
        self.widget = widgets.VBox([
            self.name,
            ac
        ])

        #Init callbacks
        self.init_callbacks()


    def update_config(self, change):
        """Update internal config with values from widgets."""
        logger.debug('Updating config')
        if self.name.value:
            self.config['name'] = self.name.value
        else:
            self.config['name'] = None

        if self.fpath.value:
            self.config['fpath'] = self.fpath.value
        else:
            self.config['fpath'] = None

        if self.base.value:
            self.config['base'] = self.base.value
        else:
            self.config['base'] = None

        if self.norm.value:
            self.config['norm'] = self.norm.value
        else:
            self.config['norm'] = None

        # Update roi spectra
        value = self.roi_spectra.value
        logger.debug('Setting roi_spectra with {}'.format(value))
        if value == []:
            self.kwargs_record.set('roi_spectra', None)
        else:
            self.kwargs_record.set('roi_spectra', self.roi_spectra.value)

        # update roi_x_pixel_spec
        value = slice(*self.roi_x_pixel_spec.value)
        self.kwargs_record.set('roi_x_pixel_spec', value)

        # Update vis_wl
        if not self.vis_wl_disabler.value:
            self.kwargs_record.set('vis_wl', self.vis_wl.value)

        if not self.pump_unpumped_disabler.value:
            logger.debug('Setting unpumped_index {}'.format(self.unpumped_index.value))
            self.kwargs_record.set('pumped_index', self.pumped_index.value)
            self.kwargs_record.set('unpumped_index', self.unpumped_index.value)
        self.config['kwargs_record'] = self.kwargs_record.value

    def update_record(self, change):
        """Use config to update properties of data"""
        logger.debug('Updating record {} with {}'.format(self.record, self.config))
        self.record.update_config(self.config.get('kwargs_record'))

    def load_data(self, change):
        """Read record from HDD and update global records obj."""
        logger.debug('Loading Data with {}'.format(self.config))
        self.record = raw_reader.import_record(self.config, records)
        records[self.config['name']] = self.record
        # Set limits according to imported data.
        self.update_widgets(change)

        # This is needed because of bad programming
        # I somehow need to make sure that roi_spectra widgets stay usable
        # After recreating the widgets
        for wdg in self.roi_spectra.widgets:
            wdg.observe(self.update_config, names='value')
            wdg.observe(self.update_record, names='value')

    def update_widgets(self, change):
        """Update widgets with values from data."""
        self.roi_spectra.value = self.record.number_of_spectra
        self.roi_x_pixel_spec.min = 0
        self.roi_x_pixel_spec.max = self.record.number_of_x_pixel
        self.roi_x_pixel_spec.value = (
            self.record.roi_x_pixel_spec.start,
            self.record.roi_x_pixel_spec.stop
        )
        self.vis_wl.value = self.record.vis_wl
        self.pumped_index.value = self.record.pumped_index
        self.unpumped_index.value = self.record.unpumped_index

    def on_disable_vis_wl(self, change):
        logger.debug('Vis WL Disable called.')
        self.vis_wl.disabled = self.vis_wl_disabler.value
        if self.vis_wl.disabled:
            try:
                self.config['kwargs_record'].pop('vis_wl')
            except KeyError:
                pass

    def on_disable_pumped_unpumped_index(self, change):
        logger.debug('Toggle Pumped/Unpumped index disabler')
        self.pumped_index.disabled = self.pump_unpumped_disabler.value
        self.unpumped_index.disabled = self.pump_unpumped_disabler.value
        if self.pump_unpumped_disabler.value:
            try:
                self.config['kwargs_record'].pop('pumped_index')
                self.config['kwargs_record'].pop('unpumped_index')
            except KeyError:
                pass

    def init_callbacks(self):
        self.load.on_click(self.load_data)
        self.vis_wl_disabler.observe(self.on_disable_vis_wl, names='value')
        self.pump_unpumped_disabler.observe(
            self.on_disable_pumped_unpumped_index, names='value'
        )
        for wdg in self._widgets:
            wdg.observe(self.update_config, names='value')
            wdg.observe(self.update_record, names='value')

        for wdg in self.roi_spectra.widgets:
            wdg.observe(self.update_config, names='value')
            wdg.observe(self.update_record, names='value')

class ReadRecords(MyWidgetBase):
    def __init__(self):
        # widgets of records added during runtime
        self._record_widgets = []
        self.wFpath = widgets.Text(
            description='Path'
        )
        self.wLoad = widgets.Button(
            description='Load',
        )
        self.wSave = widgets.Button(
            description='Save',
        )
        wPath = widgets.HBox([
            self.wFpath,
            self.wLoad,
            self.wSave,
        ])

        # List of visible widgets
        self.widgets = [wPath]
        # VBox of final widget
        self.widget = widgets.VBox(self.widgets)

        self.register_callbacks()

        self.config_records = None
        self.records = None

    def read_data_cb(self, args):
        """Read data from hdd."""
        logger.debug('Loading Records from {}'.format(self.wFpath.value))
        records = {}
        widget_list = []
        self._record_widgets = []
        self.config_records = raw_reader.read_yaml(self.wFpath.value)
        os.chdir(os.path.dirname(self.wFpath.value))
        for config in self.config_records:
            logger.debug('Importing {}'.format(config))
            #raw_reader.import_record(config, records)
            rc = Record(config)
            rc.load_data(None)
            self._record_widgets.append(rc)
            widget_list.append(rc.widget)
            records[config['name']] = rc.record
        ac = widgets.Accordion(children=widget_list)
        for i in range(len(widget_list)):
            ac.set_title(i, self.config_records[i]['name'])

        self.records = records
        self.widget.children = (self.widget.children[0], ac,)
        #self._show_config_records()

    def save_config_cb(self, args):
        """Save config dict."""
        pass
        #raw_reader.

    def register_callbacks(self):
        """define callbacks"""
        self.wLoad.on_click(self.read_data_cb)


