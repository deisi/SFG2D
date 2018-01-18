"""Module to read raw data and produce records with."""
import yaml, sys
from .core import SfgRecord, SfgRecords_from_file_list

fname = 'raw_config.yaml'

with open(fname) as ifile:
    config = yaml.load(ifile)


records = {}

def main():
    for record_entrie in config['records']:
        try:
            fpath = record_entrie['fpath']
            if type(fpath) == str:
                record = SfgRecord(fpath)
            else:
                SfgRecords_from_file_list(fpath)
            records[record_entrie['name']] = record
        except OSError:
            print('During Import of {}\nFile not found {}'.format(record_entrie['name'], fpath))
