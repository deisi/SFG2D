from numpy import loadtxt
from pandas import DataFrame, to_timedelta
from datetime import datetime, timedelta
import io

class NtbFile():
    def __init__(self, fname):
        """Read .ntb file, as they are exported by the FilmWaterX Tensiometer Software

        Tested with version: 3.62

        Properties
        -----------
        header : dict
            Dictionary with all the Header information

        Header
        ------
        Time : int
            Creation time of the file in seconds relative to 00:00 of the day of creation

        Date : str
            Date of the program start, not the date of the file creation.
            As you might think, haha this would have been to smart wouldn't it....
            WHAT THE FUCK, WHO DID THIS.
        """
        self._fname = fname
        self._filedesc = open(fname, 'r')
        self._readFile()
        self._processData()

    def _readFile(self):
        import yaml

        with self._filedesc:
            line = ''
            self.header = []
            end_cond = 'Nr;TotalArea;Area;DeltaArea;DeltaMolecules;Pressure;Tension;Mode;Time;Temp;Potential;Radioactivity'
            while end_cond  not in line:
                line = self._filedesc.readline().replace(',', '.')

                # There are empty lines in the file that must be skipped
                if line is '\n':
                    continue

                # Correct for the not yaml conform lines
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
                    self._names = yaml.load(line).split(';')
                    break

                self.header.append(yaml.load(line))

            # We want to have a single dict with the header information
            self.header = {k: v for d in self.header for k, v in d.items()}

            # Read the actual data
            data = self._filedesc.read().replace(',', '.')
            self.data = loadtxt(io.StringIO(data), delimiter=';')

    def _processData(self):
        self.df = DataFrame(self.data, columns=self._names)

        # I Want to Process the Time Column to use datetime
        # Because this makes comparisons much easier.
        # This way, time is a absolute time axis
        date = datetime.strptime(self.header["Date"], "%d.%m.%Y")
        time = timedelta(seconds=self.header["Time"])
        self.df['TimeDelta'] = to_timedelta(self.df["Time"], unit='s')
        self.df["Time"] = date + time - (self.df['TimeDelta'].iloc[-1] - self.df["TimeDelta"])
