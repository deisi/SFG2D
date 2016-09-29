"""Remove double occurence of AVG in folder given by first argument.
Second arg must be do to actualy perform the renaming."""

import os
import sys
from glob import glob
from numpy import sort

if len(sys.argv) < 2:
    print("AVG_remover input_folder [do]")
    sys.exit()

ffiles = sort(glob(sys.argv[1] + '/*'))
ffiles = [s for s in ffiles if "AVG" in s]

rename = {}
for i in range(len(ffiles)-1):
    ffile = os.path.splitext(ffiles[i])
    nffile = os.path.splitext(ffiles[i+1])
    if ffile[0] + "AVG" == nffile[0] :
        rename[ffiles[i]] = ffiles[i].replace("AVG", "", 1)
        rename[ffiles[i+1]] = ffiles[i+1].replace("AVG", "", 1)
                  
print("renaming:/n")
for key, value in rename:
    print("%s ----> %s"%(key, value))

if sys.argv[1] == "do":
    for ffile in ffiles:
        new_name = rename.get(ffile)
        if new_name:
            os.rename(ffile, new_name)
