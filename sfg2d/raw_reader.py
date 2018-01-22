"""Module to read raw data and produce records with."""
import yaml, sys
from .core import SfgRecord, SfgRecords_from_file_list

fname = 'raw_config.yaml'

with open(fname) as ifile:
    config = yaml.load(ifile)


records = {}

def main():
    for record_entrie in config['records']:
        #try:
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
                print(norm.mean())

            if type(fpath) == str:
                record = SfgRecord(fpath, **record_kwgs)
            else:
                record = SfgRecords_from_file_list(fpath, **record_kwgs)
            records[record_entrie['name']] = record
        #except OSError:
        #    print('During Import of {}\nFile not found {}'.format(record_entrie['name'], fpath))
