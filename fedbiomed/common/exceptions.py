class EnvironException(Exception):
    """
    Exception specific to the Environ class
    """
    pass

class LoggerException(Exception):
    """
    Exception specific to the Logger class
    """
    pass

class StrategyException(Exception):
    """
    Exception specific to the Strategy class and subclasses
    """
    pass


class TrainingException(Exception):
    """
    Exception raises then training (researcher/node) class
    """
    pass
