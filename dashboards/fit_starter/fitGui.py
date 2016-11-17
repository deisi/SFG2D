import os
import probfit
import iminuit
import json
from IPython.display import display
from ipywidgets import Text, FloatText, VBox, Button, HBox, Checkbox, FloatRangeSlider, Tab
from glob import glob
import matplotlib.pyplot as plt

def _limit_description_to_key(description):
    """Translate between the description of the Text widget and the corresonding
    key and value pos in the fitarg dictionary.

    Parameters
    ----------
    description : str
        The string describing the widget

    Returns
    --------
    key : string
    The key in the fitarg dictionary

    attr: (0, 1)
    The entry position in the value of the fitarg dictionary
    """
    # Splits parameter name and _min or _max off
    # attr is either min or max
    key, attr = description.split('_')
    # Add limit so its is the correct key for self.fitarg
    key = 'limit_' + key
    if attr == 'min':
        attr = 0
    elif attr == 'max':
        attr = 1
    else:
        raise NotImplementedError(
            "Uuups there is something wrong." \
            "attr was %s but min/max was expected" % attr
        )
    return key, attr

def _key_to_limit_description(key, attr):
    """Take key and attribute value from fitarg and translate to.
    Description name for the limit widgets."""
    if not 'limit_' in key:
        raise IOError('Ups this should not be called this way.')
    ret = key.split('_')[1]
    if attr == 0:
        ret += '_min'
    else:
        ret += '_max'
    return ret

    
class FitGui:
    _debug = 0

    def __init__(self, x, y, y_err, fit_func,
                 selection=slice(None, None), fig=None, ax=None, fitarg={}
    ):
        '''Gui to help finding starting parameters in an sfg fit

        Parameters
        ----------
        x : array
            x data
        y : array

        y_err : y error data

        selection : slice
            subregion of the dataset to fit.
        '''
        self.x_data = x
        self.y_data = y
        self.y_err = y_err
        self.fit_func = fit_func
        self.sl = selection
        self.ax = ax
        self.chi2 = None
        self.m = None
        self.fitarg = fitarg # dictionary for minuit starting parameters

    def __call__(self):
        self._init_widgets()
        self._init_observer()
        self._on_clicks()
        display(self.w_header_box, self.w_tabs)
        self._init_figure()

    def _set_minuit(self, *args, **kwargs):
        '''here we set the minuit starting parameters and boundary condition'''
        self.m = iminuit.Minuit(
            *args,
            **kwargs
        )
        # update fitarg to reflect the same as m.fitarg
        self.fitarg = self.m.fitarg

    def _set_chi2(self):
        """Chi2 is the property we minimize"""
        self.chi2 = probfit.Chi2Regression(
            self.fit_func,
            self.x_data[self.sl],
            self.y_data[self.sl],
            self.y_err[self.sl],
        )

    def _init_widgets(self):
        """Dynamically create boxes to enter fit parameters"""
        self._set_chi2()
        self._set_minuit(self.chi2, pedantic=False, **self.fitarg)

        # we want one widget per fit value
        self._w_values = []
        for key in self.m.parameters:
            value = self.m.fitarg[key]
            self._w_values.append(
                FloatText(value=value, description=key)
            )
        self.w_value_box = VBox(self._w_values)

        # one 2 FloatText widgets per fit value limit
        self._w_limits = []
        for key in self.m.parameters:
            limit_key = 'limit_' + key
            self._w_limits.append(
                HBox([
                    FloatText(description=key+'_min'),
                    FloatText(description=key+'_max')
                    ])
            )
        self.w_limit_box = VBox(self._w_limits)

        # Check boxes to fix parameters
        self._w_fix = []
        for key in self.m.parameters:
            key = 'fix_' + key
            self._w_fix.append(
                Checkbox(description=key)
            )
        self.w_fix_box = VBox(self._w_fix)

        # A button to start the fit
        self.w_fit_button = Button(
            description='Fit',
            tooltip='Run the fitting routine',
        )

        # A button to save the fit results
        self.w_save_button = Button(
            description='save',
            tooltip='Save fit settings to file',
        )

        # A button to load the fit results
        self.w_load_button = Button(
            description='load',
            tooltip='Load fit settings from file',
        )

        # Test box for save/load file
        self.w_fitargs_path = Text(
            description='path',
            tooltip='path to save/load fit settings/nTipp: you can copy past from explorer',
            value='./starting_values.txt',
        )

        # The part above the tabs, that is always visible
        self.w_header_box = HBox([
            self.w_fit_button,
            self.w_fitargs_path,
            self.w_load_button,
            self.w_save_button,
        ])

        # The content of the first tab page
        self.w_box = HBox([
            self.w_value_box,
            self.w_fix_box,
            #self.w_limit_box,
            #self.w_fit_button
        ])

        # The content of the tabs
        self.w_tabs = Tab(children=[
            self.w_box,
            self.w_limit_box
        ])

    def _on_clicks(self):
        self.w_fit_button.on_click(self._run_fit)
        self.w_save_button.on_click(self._save_button_clicked)
        self.w_load_button.on_click(self._load_button_clicked)


    def _init_observer(self):
        for child in self.w_value_box.children:
            child.observe(self._starting_values_observer, 'value')
            child.observe(self._start_fit_line_observer, 'value')

        for child in self.w_fix_box.children:
            child.observe(self._fix_box_observer, 'value')

        for box in self.w_limit_box.children:
            for child in box.children:
                child.observe(self._parameter_range_observer, 'value')

    def _unobserve(self):
        for child in self.w_value_box.children:
            child.unobserve_all()

        for child in self.w_fix_box.children:
            child.unobserve_all

        for box in self.w_limit_box.children:
            for child in box.children:
                child.unobserve_all()

    def _init_figure(self):
        if not self.ax:
            self.ax = plt.gca()

        plt.plot(self.x_data, self.y_data)
        self._fit_line = plt.plot(self.chi2.x, self.chi2.f(self.chi2.x, *self.m.args))

    #########################################
    # _update functions
    ########################################
    # TODO remove the new args and make observer callback function

    def _update_gui(self):
        """Take changes in self.fitargs and refelct in the gui"""
        self._unobserve()
        for child in self.w_value_box.children:
            new_value = self.fitarg[child.description]
            child.value =  new_value

        for child in self.w_fix_box.children:
            new_value = self.fitarg[child.description]
            child.value = new_value

        for box in self.w_limit_box.children:
            for child in box.children:
                fitargs_key, attr = _limit_description_to_key(child.description)
                new_value = self.fitarg[fitargs_key]
                if isinstance(new_value, type(None)):
                    continue
                child.value = new_value[attr]

        self._update_minuit()
        self._init_observer()
        self._update_start_fit_line()

    def _update_minuit(self):
        """Use self.fitarg to update minuit"""
        self._set_minuit(self.chi2, pedantic=False, **self.fitarg)


    def _update_starting_values(self):
        for child in self.w_value_box.children:
            key = child.description
            self.fitarg[key] = child.value
        self._set_minuit(self.chi2, pedantic=False, **self.fitarg)

    def _run_fit(self, new):
        if self._debug:
            print("_run_fit was called")
        self.fres = self.m.migrad()

        # we want the input widgets to reflect the new results from
        # the fit. So we need to update them according to the results
        new_parameter_values = {}
        for parameter_result_dict in self.fres[1]:
            # That's the name of the current parameter
            key = parameter_result_dict['name']
            value = parameter_result_dict['value']
            new_parameter_values[key] = value

        for child in self.w_value_box.children:
            key = child.description
            child.value = new_parameter_values[key]
            if self._debug > 1:
                print('updating %s with %s' % (key, child.value))

    def _update_start_fit_line(self):
        '''update the initial fit value representation'''
        if self._debug:
            print('Update_start_fit_line called')
        y_fit_start = self.chi2.f(self.chi2.x, *self.m.args)
        self._fit_line[0].set_data(
            self.chi2.x,
            y_fit_start
        )
        self.ax.figure.canvas.draw()


    ###################################################
    # observer callback function
    ###################################################

    def _start_fit_line_observer(self, new):
        self._update_start_fit_line()

    def _starting_values_observer(self, new):
        self._update_starting_values()

    def _parameter_range_observer(self, new):
        """This got a bit more complecated, because the Ranges depend on each other"""
        widget = new["owner"]
        key, attr = _limit_description_to_key(widget.description)

        # update value 
        value = self.fitarg[key]
        if isinstance(value, type(None)):
            value = (0, 1)
        value = list(value)
        value[attr] = widget.value
        self.fitarg[key] = tuple(value)

        # only update minuit if value is reasonable
        if value[1] > value[0]:
            if self._debug:
                print('updating minuit called by _parameter_range_observer')
            self._set_minuit(self.chi2, pedantic=False, **self.fitarg)

    def _fix_box_observer(self, new):
        if self._debug:
            print('update_parameter_fix called')
        if self._debug > 1:
            print('new is %s' % new)
        # Select correct widget and update fitargs accordingly
        owner = new['owner']
        key = owner.description
        self.fitarg[key] = owner.value
        self._set_minuit(self.chi2, pedantic=False, **self.fitarg)

    def _save_button_clicked(self, new):
        fp = self.w_fitargs_path.value
        self.save(fp)

    def _load_button_clicked(self, new):
        if self._debug > 0:
            print("load button clicked")
        fp = self.w_fitargs_path.value
        self.fitarg = self.load(fp)
        self._update_gui() # show changes in the gui
        #self._update_starting_values()
        #self._fix_box_observer(None)

    def save(self, fp):
        """Save the result of the fit into a text file"""
        with open(fp, "w") as outfile:
            json.dump(self.fitarg, outfile, indent=4, separators=(',', ': '), sort_keys=True)

    def load(self, fp):
        """Load fitargs from the text file fp"""
        with open(fp) as json_data:
            ret = json.load(json_data)
            if self._debug > 2:
                print('Read %s from %s' % (ret, fp))
        return ret


