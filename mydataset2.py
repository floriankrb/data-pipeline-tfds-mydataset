"""mydataset dataset."""

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from mydataset_generic import MydatasetGeneric
class Mydataset2(MydatasetGeneric):
    SHAPE = (46, 121 // 20 + 1, 240 // 20)
    SUBSAMPLE = 20
    dev = True
