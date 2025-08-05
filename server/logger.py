# used to check all kind of warning,information and error messages that we might need for debugging
import logging


def setup_logger(name="ragbot"):

    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG) #what reason we want to use log file -> that is debugging

    # Console handler
    ch=logging.StreamHandler() #send and show all the logs to console
    ch.setLevel(logging.DEBUG) # set the level of the console handler to debug

    # formatter (in which format we want to see the logs)
    formatter=logging.Formatter("[%(asctime)s] [%(levelname)s] -  %(message)s ")
    ch.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)

    return logger





logger=setup_logger()