import unittest

import threading
import time
import sys

import testsupport.mock_researcher_environ
from   testsupport.broker import FakeBroker

from fedbiomed.researcher.environ import MQTT_BROKER, MQTT_BROKER_PORT
from fedbiomed.common.messaging   import Messaging, MessagingType

class TestMessaging(unittest.TestCase):
    '''
    Test the Messaging class
    '''
    # once in test lifetime
    @classmethod
    def setUpClass(cls):

        # start a fake broker in a separate thread
        try:
            cls.broker = FakeBroker(port = MQTT_BROKER_PORT)
            print(cls.broker.__dict__)
            thread = threading.Thread(target = cls.broker.start, args = () )

        except Exception as e:
            print("cannot start FakeBroker:", e)

        time.sleep(1.0) # give some time to the thread to listen to the given port
        pass

    @classmethod
    def tearDownClass(cls):
        cls.broker.finish()
        pass

    # before all the tests
    def setUp(self):
        pass

    # after all the tests
    def tearDown(self):
        pass

    # mqqt callbacks
    def on_message(msg):
        print("RECV:", msg)

    # tests
    def test_connect(self):
        '''
        messaging constructor test
        '''
        print("PORT=", MQTT_BROKER_PORT)
        print("PORT=", self.broker._port)
        print("test_connect")

        i = 0
        while i < 100:
            time.sleep(1)
            if self.broker._conn is not None:
                print("receiving")
                self.broker.receive_packet( 20 )
            else:

                print("conn is None")
                print(self.broker.__dict__)
            i += 1
        #m = Messaging(self.on_message,
        #              MessagingType.RESEARCHER,
        #              "XX-researcher",
        #              "localhost",
        #              1888)

        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
