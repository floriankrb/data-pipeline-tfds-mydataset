"""mydataset dataset."""

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


from mydataset_generic import MydatasetGeneric


class Mydataset(MydatasetGeneric):
    SHAPE = (46, 121, 240)
    SUBSAMPLE = False
    dev = False
