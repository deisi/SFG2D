"""Module to read raw data and produce records with."""
import yaml, sys
from .core import SfgRecord

fname = 'raw_config.yaml'

with open(fname) as ifile:
    config = yaml.load(ifile)


records = {}
for record_entrie in config['records']:
    records[record_entrie['name']] = SfgRecord(record_entrie['fpath'])
