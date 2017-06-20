from analysis_engine.library import (
    any_deps,
    vstack_params_where_state,
)

from analysis_engine.node import (
    M,
    MultistateDerivedParameterNode,
)


class GearDown(MultistateDerivedParameterNode):

    values_mapping = {
        0: 'Up',
        1: 'Down',
    }

    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        return any_deps(cls, available)

    def derive(self,
               gl=M('Gear (L) Down'),
               gn=M('Gear (N) Down'),
               gr=M('Gear (R) Down'),
               gc=M('Gear (C) Down')):
        # Join all available gear parameters and use whichever are available.
        self.array = vstack_params_where_state(
            (gl, 'Down'),
            (gn, 'Down'),
            (gr, 'Down'),
            (gc, 'Down'),
        ).all(axis=0)


class GearUp(MultistateDerivedParameterNode):

    values_mapping = {
        0: 'Down',
        1: 'Up',
    }

    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        return any_deps(cls, available)

    def derive(self,
               gl=M('Gear (L) Up'),
               gn=M('Gear (N) Up'),
               gr=M('Gear (R) Up'),
               gc=M('Gear (C) Up')):
        # Join all available gear parameters and use whichever are available.
        self.array = vstack_params_where_state(
            (gl, 'Up'),
            (gn, 'Up'),
            (gr, 'Up'),
            (gc, 'Up'),
        ).all(axis=0)


class GearInTransit(MultistateDerivedParameterNode):

    values_mapping = {
        0: '-',
        1: 'In Transit',
    }

    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        return any_deps(cls, available)

    def derive(self,
               gl=M('Gear (L) In Transit'),
               gn=M('Gear (N) In Transit'),
               gr=M('Gear (R) In Transit'),
               gc=M('Gear (C) In Transit')):
        # Join all available gear parameters and use whichever are available.
        self.array = vstack_params_where_state(
            (gl, 'In Transit'),
            (gn, 'In Transit'),
            (gr, 'In Transit'),
            (gc, 'In Transit'),
        ).any(axis=0)


class GearDownInTransit(MultistateDerivedParameterNode):

    values_mapping = {
        0: '-',
        1: 'Extending',
    }

    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        return any_deps(cls, available)

    def derive(self,
               gear_L=M('Gear (L) Down In Transit'),
               gear_N=M('Gear (N) Down In Transit'),
               gear_R=M('Gear (R) Down In Transit'),
               gear_C=M('Gear (C) Down In Transit')):
        combine_params = [(x, 'Extending') for x in (gear_L, gear_R, gear_N, gear_C) if x]
        if len(combine_params):
            self.array = vstack_params_where_state(*combine_params).any(axis=0)


class GearUpInTransit(MultistateDerivedParameterNode):

    values_mapping = {
        0: '-',
        1: 'Retracting',
    }

    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        return any_deps(cls, available)

    def derive(self,
               gear_L=M('Gear (L) Up In Transit'),
               gear_N=M('Gear (N) Up In Transit'),
               gear_R=M('Gear (R) Up In Transit'),
               gear_C=M('Gear (C) Up In Transit')):
        combine_params = [(x, 'Retracting') for x in (gear_L, gear_R, gear_N, gear_C) if x]
        if len(combine_params):
            self.array = vstack_params_where_state(*combine_params).any(axis=0)
