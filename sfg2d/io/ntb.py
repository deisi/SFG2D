from numpy import genfromtxt
import yaml


class NtbFile():
    def __init__(self, fname):
        self._fname = fname
        self._filedesc = open(fname, 'r')
        self._readHeader()
        self._readData()

    def _readHeader(self):
        with self._filedesc:
            line = ''
            self.header = []
            end_cond = 'Nr;TotalArea;Area;DeltaArea;DeltaMolecules;Pressure;Tension;Mode;Time;Temp;Potential;Radioactivity'
            while end_cond  not in line:
                line = self._filedesc.readline().replace(',', '.')

                # There are empty lines in the file that must be skipped
                if line is '\n':
                    continue

                # Correct for the not yaml comform lines
                if 'Lipid(s) Details' in line:
                    lipid = yaml.load(line)
                    lipid['Lipid(s) Details'] = yaml.load(
                        self._filedesc.readline().replace(',', '.')
                    )
                    lipid['Lipid(s) Details'] += yaml.load(
                        self._filedesc.readline().replace(',', '.')
                    )
                    self.header.append(lipid)
                    continue

                if end_cond in line:
                    self.names = yaml.load(line).split(';')
                    break

                self.header.append(yaml.load(line))

            self.end_of_header = self._filedesc.tell()
            # Combine the list od dictionaries
            self.header = {k: v for d in self.header for k, v in d.items()}

    def _readData(self):
        return
        self._data = genfromtxt(
            self._fname, delimiter=';', skip_header=87, names=True,
            skip_footer=1
        )
