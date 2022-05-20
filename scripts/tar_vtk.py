#!/usr/bin/env python

import sys
import pyathena as pa

print(sys.argv)
s = pa.LoadSimTIGRESSNCR(sys.argv[1],verbose=True)
s.create_tar_all(kind='vtk',remove_original=True)
