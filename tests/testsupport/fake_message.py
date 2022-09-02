""" This file contains dummy Classes for unit testing. It fakes NodeMessages
(from fedbiomed.common.messages) 
"""

from typing import Any, Dict


class FakeMessages:
    """Fakes NodeMessages class. 
    Provides a constructor and a `get_item` method
    """
    def __init__(self, msg: Dict[str, Any]):
        """Constructor of dummy class NodeMessages

        Args:
            msg (Dict[str, Any]): a message (can be any dictionary)
        """
        self.msg = msg

    def get_dict(self) -> Dict[str, Any]:
        """Methods that returns the msg stored in the class

        Returns:
            Dict[str, Any]: returns the message stored in class
        """
        return self.msg
    
    def get_param(self, val: str) -> Any:
            return self.msg.get(val)
