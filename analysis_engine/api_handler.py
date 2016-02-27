# -*- coding: utf-8 -*-
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
##############################################################################

'''
'''

##############################################################################
# Imports


import logging
import os
import requests
import sys

from requests.packages.urllib3.util.retry import Retry


##############################################################################
# Globals


logger = logging.getLogger(name=__name__)


if getattr(sys, 'frozen', False):
    # XXX: Attempt to provide path to certificates in frozen applications:
    path = os.path.join(os.path.dirname(sys.executable), 'cacert.pem')
    os.environ.setdefault('REQUESTS_CA_BUNDLE', path)


##############################################################################
# Exceptions


class APIError(Exception):
    '''
    A generic exception class for an error when calling an API.
    '''

    def __init__(self, message, url=None, method=None, params=None, data=None, json=None):
        super(APIError, self).__init__(message)
        self.url = url
        self.method = method
        self.params = params
        self.data = data
        self.json = json


class IncompleteEntryError(APIError):
    '''
    An exception to be raised when and entry does not contain all required
    data.
    '''
    pass


class NotFoundError(APIError):
    '''
    An exception to be raised when something could not be found via the API.
    '''
    pass


##############################################################################
# HTTP API Handler


class APIHandlerHTTP(object):
    '''
    Restful HTTP API Handler.
    '''

    def request(self, url, method='GET', params=None, data=None, json=None, **kw):
        '''
        Makes a request to a URL and attempts to return the decoded content.

        :param url: url to connect to for handling the request.
        :type url: str
        :param method: method for the request.
        :type method: str
        :param data: data to send in the body of the request.
        :type data: mixed
        :returns: the data fetched from the remote server.
        :rtype: mixed
        :raises: NotFoundError -- if no record could be found (server returns 404)
        :raises: APIError -- if the server does not respond or returns an error.
        '''
        backoff = kw.get('backoff', 0.2)
        retries = kw.get('retries', 5)
        timeout = kw.get('timeout', 15)

        retries = Retry(total=retries, backoff_factor=backoff, status_forcelist=[503])

        logger.debug('API Request: %s %s', method, url)
        try:
            with requests.Session() as s:
                a = requests.adapters.HTTPAdapter(max_retries=retries)
                s.mount('http://', a)
                s.mount('https://', a)
                r = s.request(method, url, params=params, data=data, json=json, timeout=timeout)
                r.raise_for_status()
                return r.json()

        except requests.HTTPError as e:
            try:
                message = e.response.json()['error']
            except:
                message = 'No error message available or supplied.'
            if e.response.status_code == requests.codes.not_found:
                logger.debug(message)
                raise NotFoundError(message, url, method, params, data, json)
            else:
                logger.exception(message)
                raise APIError(message, url, method, params, data, json)
        except requests.RequestException:
            message = 'Unexpected error with connection to the API.'
            logger.exception(message)
            raise APIError(message, url, method, params, data, json)
        except ValueError:
            # Note: JSONDecodeError only in simplejson or Python 3.5+
            message = 'Unexpected error decoding response from API.'
            logger.exception(message)
            raise APIError(message, url, method, params, data, json)
        except:
            message = 'Unexpected error from the API.'
            logger.exception(message)
            raise APIError(message, url, method, params, data, json)


##############################################################################
# API Handler Lookup Function


def get_api_handler(handler_path, *args, **kwargs):
    '''
    Returns an instance of the class specified by the handler_path.

    :param handler_path: Path to handler module, e.g. project.module.APIHandler
    :type handler_path: string
    :param args: Handler class instantiation args.
    :type args: list
    :param kwargs: Handler class instantiation kwargs.
    :type kwargs: dict
    '''
    import_path_split = handler_path.split('.')
    class_name = import_path_split.pop()
    module_path = '.'.join(import_path_split)
    handler_module = __import__(module_path, globals(), locals(),
                                fromlist=[class_name])
    handler_class = getattr(handler_module, class_name)
    return handler_class(*args, **kwargs)
