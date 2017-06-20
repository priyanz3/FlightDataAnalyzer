# -*- coding: utf-8 -*-

import numpy as np

from flightdatautilities import units as ut

from analysis_engine.library import any_deps, blend_two_parameters
from analysis_engine.node import DerivedParameterNode, P


class Groundspeed(DerivedParameterNode):
    
    align = False
    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return any_deps(cls, available)

    def derive(self,
               # aeroplane
               source_A=P('Groundspeed (1)'),
               source_B=P('Groundspeed (2)')):
        self.array, self.frequency, self.offset = blend_two_parameters(source_A, source_B)