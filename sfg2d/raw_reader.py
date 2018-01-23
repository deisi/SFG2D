"""Module to read raw data and produce records with."""
import yaml, sys
import sfg2d.core as core
import sfg2d.fig as fig

fname = 'raw_config.yaml'


def main():
    global records, figures, config
    records = {}
    figures = {}
    config = {}

    with open(fname) as ifile:
        config = yaml.load(ifile)
    for record_entrie in config['records']:
        fpath = record_entrie['fpath']
        record_kwgs = record_entrie.get('record_kwgs', {})
        base_dict = record_entrie.get('base')
        if base_dict:
            base = records[base_dict['name']].select(
                prop='rawData',
                frame_med=True
            )
            record_kwgs['base'] = base

        norm_dict = record_entrie.get('norm')
        if norm_dict:
            norm = records[norm_dict['name']].select(
                prop='basesubed',
                frame_med=True
            )
            record_kwgs['norm'] = norm

        if type(fpath) == str:
            record = core.SfgRecord(fpath, **record_kwgs)
        else:
            record = core.SfgRecords_from_file_list(fpath, **record_kwgs)
        records[record_entrie['name']] = record

    for fig_config in config['figures']:
        # Name is equal the config key, so it must be stripped
        fig_name, fig_config = list(fig_config.items())[0]
        fig_func = getattr(fig, fig_config['type'])
        fig_kwgs = fig_config['fig_kwgs'].copy()
        record = fig_kwgs.get('record')
        if record:
            fig_kwgs['record'] = records[record]
        print(fig_name)
        print('select_kw: ', fig_kwgs['select_kw'])
        print('###############')
        figures[fig_name] = fig_func(**fig_kwgs)
