"""Implement an ESPEILogger with customized logging levels and methods"""

import logging

class ESPEILogger(logging.getLoggerClass()):
    TRACE = 15
    def trace(self, *args, **kwargs):
        self.log(self.TRACE, *args, **kwargs)

def _setup_logging():
    logging.addLevelName(15, 'TRACE')
    logging.setLoggerClass(ESPEILogger)