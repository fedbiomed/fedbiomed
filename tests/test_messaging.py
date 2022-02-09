# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ
# Import environ for researcher, since tests will be running for researcher component
from fedbiomed.researcher.environ import environ

import unittest
import random
import sys
import threading
import time

from fedbiomed.common.messaging      import Messaging
from fedbiomed.common.constants      import ComponentType
from fedbiomed.common.message        import ResearcherMessages
from fedbiomed.common.message        import NodeMessages


class TestMessaging(unittest.TestCase):
    '''
    Test the Messaging class connect/disconnect
    '''

    def test_messaging_00_bad_init(self):

        self._m = Messaging(on_message       = None,
                            messaging_type   = None,
                            messaging_id     = 1234,
                            mqtt_broker      = "1.2.3.4",
                            mqtt_broker_port = 1)

        self.assertFalse( self._m.is_connected )
        self.assertEqual( self._m.default_send_topic, None)



class TestMessagingResearcher(unittest.TestCase):
    '''
    Test the Messaging class from the researcher point of view
    '''

    # once in test lifetime
    @classmethod
    def setUpClass(cls):

        random.seed()
        # verify that a broker is available
        try:
            print("connecting to:", environ['MQTT_BROKER'], "/", environ['MQTT_BROKER_PORT'])
            cls._m = Messaging(cls.on_message,
                               ComponentType.RESEARCHER,
                               environ['RESEARCHER_ID'],
                               environ['MQTT_BROKER'],
                               environ['MQTT_BROKER_PORT']
                            )

            cls._m.start()
            cls._broker_ok = True
        except Exception as e:
            cls._broker_ok = False

        time.sleep(1.0) # give some time to the thread to listen to the given port
        print("MQTT connexion status =" , cls._broker_ok)

        pass


    @classmethod
    def tearDownClass(cls):
        if not cls._broker_ok:
            cls._m.stop()
        pass


    # before all the tests
    def setUp(self):
        pass


    # after all the tests
    def tearDown(self):
        pass


    # mqqt callbacks
    def on_message(msg, topic):
        print("RECV:", topic, msg)


    # tests
    def test_messaging_researcher_00_init(self):

        self.assertEqual( self._m.default_send_topic, "general/nodes")
        self.assertEqual( self._m.on_message_handler, TestMessagingResearcher.on_message)

    def test_messaging_researcher_01_send(self):
        '''
        send a message on MQTT
        '''
        if not self._broker_ok:
            self.skipTest('no broker available for this test')

        try:
            ping = ResearcherMessages.request_create(
                {'researcher_id' : environ['RESEARCHER_ID'],
                 'sequence'      : random.randint(1, 65535),
                 'command'       :'ping'}).get_dict()
            self._m.send_message(ping)
            self.assertTrue( True, "ping message correctly sent")

        except:
            self.assertTrue( False, "ping message not sent")


class TestMessagingNode(unittest.TestCase):
    '''
    Test the Messaging class from the Node point of view
    '''

    # once in test lifetime
    @classmethod
    def setUpClass(cls):

        random.seed()
        # verify that a broker is available
        try:
            print("connecting to:", environ['MQTT_BROKER'], "/", environ['MQTT_BROKER_PORT'])
            cls._m = Messaging(cls.on_message,
                               ComponentType.NODE,
                               environ['NODE_ID'],
                               environ['MQTT_BROKER'],
                               environ['MQTT_BROKER_PORT']
                            )

            cls._m.start()
            cls._broker_ok = True
        except Exception as e:
            cls._broker_ok = False

        time.sleep(1.0) # give some time to the thread to listen to the given port
        print("MQTT connexion status =" , cls._broker_ok)

        pass

    @classmethod
    def tearDownClass(cls):
        if not cls._broker_ok:
            cls._m.stop()
        pass


    # before all the tests
    def setUp(self):
        pass


    # after all the tests
    def tearDown(self):
        pass


    # mqqt callbacks
    def on_message(msg, topic):
        print("RECV:", topic, msg)


    # tests
    def test_messaging_node_00_init(self):

        self.assertEqual( self._m.default_send_topic, "general/researcher")
        self.assertEqual( self._m.on_message_handler, TestMessagingNode.on_message)

    def test_messaging_node_01_send(self):
        '''
        send a message on MQTT
        '''
        if not self._broker_ok:
            self.skipTest('no broker available for this test')

        try:
            ping = NodeMessages.reply_create(
                {'researcher_id' : 'XXX',
                 'node_id'       : environ['NODE_ID'],
                 'sequence'      : random.randint(1, 65535),
                 'success'       : True,
                 'command'       :'pong'}).get_dict()
            self._m.send_message(ping)
            self.assertTrue( True, "pong message correctly sent")

        except:
            self.assertTrue( False, "pong message not sent")



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
