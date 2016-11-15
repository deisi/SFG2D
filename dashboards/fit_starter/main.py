import os
import probfit
import iminuit
import pandas 
from IPython.display import display
from ipywidgets import Text, FloatText, VBox, Button, HBox, Checkbox, FloatRangeSlider, Tab
from glob import glob
import matplotlib.pyplot as plt

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
        # The combined widget box to display
        self.w_box = HBox([
            self.w_value_box,
            self.w_fix_box,
            #self.w_limit_box,
            self.w_fit_button
        ])

        self.w_tabs = Tab(children=[
            self.w_box,
            self.w_limit_box
        ])

    def _update_parameter_range(self, new):
        """This got a bit more complecated, because the Ranges depend on each other"""
        widget = new["owner"]
        # Splits parameter name and _min or _max off
        # attr is either min or max
        key, attr = widget.description.split('_')
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
                print('updating minuit called by _update_parameter_range')
            self._set_minuit(self.chi2, pedantic=False, **self.fitarg)

    def _update_starting_values(self, new):
        for child in self.w_value_box.children:
            key = child.description
            self.fitarg[key] = child.value
        self._set_minuit(self.chi2, pedantic=False, **self.fitarg)

    def _run_fit(self, new):
        if self._debug:
            print("_fun_fit was called")
        self.m.migrad()

        # we want the input widgets to reflect the new results from
        # the fit
        for child in self.w_value_box.children:
            key = child.description
            child.value = self.m.fitarg[key]

    def _update_parameter_fix(self, new):
        if self._debug:
            print('update_parameter_fix called')
        if self._debug > 1:
            print('new is %s' % new)
        # Select correct widget and update fitargs accordingly
        owner = new['owner']
        key = owner.description
        self.fitarg[key] = owner.value
        self._set_minuit(self.chi2, pedantic=False, **self.fitarg)

    def _init_observer(self):
        for child in self.w_value_box.children:
            child.observe(self._update_starting_values, 'value')
            child.observe(self._update_start_fit_line, 'value')

        self.w_fit_button.on_click(self._run_fit)

        for child in self.w_fix_box.children:
            child.observe(self._update_parameter_fix, 'value')

        for box in self.w_limit_box.children:
            for child in box.children:
                child.observe(self._update_parameter_range, 'value')

    def _init_figure(self):
        if not self.ax:
            self.ax = plt.gca()

        plt.plot(self.x_data, self.y_data)
        self._fit_line = plt.plot(self.chi2.x, self.chi2.f(self.chi2.x, *self.m.args))

    def _update_start_fit_line(self, new):
        '''update the initial fit value representation'''
        if self._debug:
            print('Update_start_fit_line called')
        y_fit_start = self.chi2.f(self.chi2.x, *self.m.args)
        self._fit_line[0].set_data(
            self.chi2.x,
            y_fit_start
        )
        self.ax.figure.canvas.draw()

    def __call__(self):
        self._init_widgets()
        self._init_observer()
        display(self.w_tabs)
        self._init_figure()



