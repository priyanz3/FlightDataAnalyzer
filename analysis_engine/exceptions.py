# -*- coding: utf-8 -*-
################################################################################

'''
Custom exceptions raised by the flight data analyser.
'''

################################################################################
# Exceptions


# NOTE: Take care when creating new exceptions. They most likely will need to
#       be pickleable to prevent errors when running the analysis engine
#       through celery.
#
# http://docs.celeryproject.org/en/latest/userguide/tasks.html#creating-pickleable-exceptions


class AFRMissmatchError(Exception):
    '''
    Error handling for cases where provided AFR does not match analysed Flight Data.
    '''

    def __init__(self, node_name, message):
        '''
        :param node_name: Node where discrepancy from AFR is detected.
        :type node_name: string
        :param message: Message explaining discrepancy.
        :type message: string
        '''
        self.node_name = node_name
        self.message = message
        super(AFRMissmatchError, self).__init__(node_name, message)

    def __str__(self):
        '''
        '''
        return "A descrepancy was detected between the provided AFR and flight data while"\
               " processing '%s': %s"\
            % (self.node_name, self.message)


class DataFrameError(Exception):
    '''
    Error handling for cases where new LFLs require a derived parameter
    algorithm which has not yet been programmed.
    '''

    def __init__(self, param_name, frame_name):
        '''
        :param param_name: Recorded parameter where frame specific processing
                           is required.
        :type param_name: string
        :param frame_name: Data frame identifier. For this parameter, no frame
                           condition evaluates true.
        :type frame_name: string
        '''
        self.param_name = param_name
        self.frame_name = frame_name
        super(DataFrameError, self).__init__(param_name, frame_name)

    def __str__(self):
        '''
        '''
        return "No valid data for parameter '%s' and no procedure for frame '%s'." \
               "\n This may be an LFL omission or caused by failing data validation checks."\
            % (self.param_name, self.frame_name)


################################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
