"""Implement an ESPEILogger with customized logging levels and methods"""

import logging

class ESPEILogger(logging.getLoggerClass()):
    TRACE = 15
    def trace(self, *args, **kwargs):
        self.log(self.TRACE, *args, **kwargs)


_VERBOSITY_LOG_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: ESPEILogger.TRACE,
    3: logging.DEBUG,
}


def _setup_logging():
    logging.addLevelName(15, "TRACE")
    logging.setLoggerClass(ESPEILogger)


def config_logger(verbosity=0, filename=None, reset_handlers=True):
    """Configure the root logger with the appropriate level and filename.
    
    Uses ESPEI's verbosity levels:
    
    * 0: Warning
    * 1: Info
    * 2: Trace
    * 3: Debug

    If the ``filename`` is None, logs will be output to stderr.
    """
    # Get the top level ESPEI logger inside this function rather than idiomatically
    # at the module level so we can be sure it has been configured.
    root_logger = logging.getLogger()

    root_logger.setLevel(_VERBOSITY_LOG_LEVELS[verbosity])

    if reset_handlers:
        root_logger.handlers.clear()
    
    if filename is not None:
        handler = logging.FileHandler(filename)
    else:
        handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s - %(message)s")
    handler.setFormatter(formatter)
    handler.addFilter(logging.Filter('espei'))
    root_logger.addHandler(handler)
