from analysis_engine.library import (
    any_deps,
    any_of,
    first_valid_parameter,
    nearest_neighbour_mask_repair,
    np_ma_masked_zeros_like,
    repair_mask,
    runs_of_ones,
    slices_remove_small_gaps,
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
        # remove any spikes
        _slices = runs_of_ones(self.array == 'Down')
        _slices = slices_remove_small_gaps(_slices, 2, self.hz)
        for _slice in _slices:
            self.array[_slice] = 'Down'        
        


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


class GearPosition(MultistateDerivedParameterNode):

    align = False
    values_mapping = {
        0: '-',
        1: 'Up',
        2: 'In Transit',
        3: 'Down',
    }


    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        merge_position = any_of(('Gear (L) Position', 'Gear (N) Position',
                                 'Gear (R) Position', 'Gear (C) Position'),
                                available)
        return merge_position

    def derive(self,
               gl=M('Gear (L) Position'),
               gn=M('Gear (N) Position'),
               gr=M('Gear (R) Position'),
               gc=M('Gear (C) Position')):
        up_state = vstack_params_where_state(
            (gl, 'Up'),
            (gn, 'Up'),
            (gr, 'Up'),
            (gc, 'Up'),
        ).all(axis=0)
        down_state = vstack_params_where_state(
            (gl, 'Down'),
            (gn, 'Down'),
            (gr, 'Down'),
            (gc, 'Down'),
        ).all(axis=0)
        transit_state = vstack_params_where_state(
            (gl, 'In Transit'),
            (gn, 'In Transit'),
            (gr, 'In Transit'),
            (gc, 'In Transit'),
        ).any(axis=0)
        param = first_valid_parameter(gl, gn, gr, gc)
        self.array = np_ma_masked_zeros_like(param.array)
        self.array[repair_mask(up_state, repair_duration=None)] = 'Up'
        self.array[repair_mask(down_state, repair_duration=None)] = 'Down'
        self.array[repair_mask(transit_state, repair_duration=None)] = 'In Transit'
        self.array = nearest_neighbour_mask_repair(self.array)
