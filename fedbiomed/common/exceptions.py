class FedbiomedException(Exception):
    """
    top class of all our exceptions

    permit to catch everything in one except:
    """
    pass

class EnvironException(FedbiomedException):
    """
    Exception specific to the Environ class
    """
    pass

class LoggerException(FedbiomedException):
    """
    Exception specific to the Logger class
    """
    pass

class MessageException(FedbiomedException):
    """
    Exception specific to the Message class
    usually a badly formed message
    """
    pass

class StrategyException(FedbiomedException):
    """
    Exception specific to the Strategy class and subclasses
    """
    pass


class TrainingException(FedbiomedException):
    """
    Exception raises then training (researcher/node) class
    """
    pass
