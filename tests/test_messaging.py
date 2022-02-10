# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ
# Import environ for researcher, since tests will be running for researcher component
from fedbiomed.researcher.environ import environ

import unittest
import sys
import threading
import time

from fedbiomed.common.messaging      import Messaging
from fedbiomed.common.constants      import ComponentType
from fedbiomed.common.message        import PingReply
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


    def test_messaging_01_researcher_init(self):

        self._m = Messaging(on_message       = None,
                            messaging_type   = ComponentType.RESEARCHER,
                            messaging_id     = 1234,
                            mqtt_broker      = "1.2.3.4",
                            mqtt_broker_port = 1)

        self.assertFalse( self._m.is_connected )
        self.assertEqual( self._m.default_send_topic, "general/nodes")

    def test_messaging_01_node_init(self):

        self._m = Messaging(on_message       = None,
                            messaging_type   = ComponentType.NODE,
                            messaging_id     = 1234,
                            mqtt_broker      = "1.2.3.4",
                            mqtt_broker_port = 1)

        self.assertFalse( self._m.is_connected )
        self.assertEqual( self._m.default_send_topic, "general/researcher")


class TestMessagingResearcher(unittest.TestCase):
    '''
    Test the Messaging class from the researcher point of view
    '''

    # once in test lifetime
    @classmethod
    def setUpClass(cls):

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

    @classmethod
    def on_message(cls, msg, topic):
        # classmethod necessary to have access to self via cls
        print("RESH_RECV:", topic, msg)

        # verify the channel
        cls.assertTrue(cls, topic, "general/researcher")

        # verify the received msg only if sent by ourself
        if  msg['researcher_id'] == 'TEST_MESSAGING_RANDOM_ID_6735424425_DO_NOT_USE_ELSEWHERE':
            cls.assertTrue(cls, msg['node_id'], "node_1234")
            cls.assertTrue(cls, msg['success'], True)
            cls.assertTrue(cls, msg['sequence'], 12345)
            cls.assertTrue(cls, msg['command'], 'pong')


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
            # here we create a PingReply as a node
            # 1/ use NodeMessages.reply_create
            # 2/ cheat with the default channel to send a msg to ourself
            # (if we don't do that, the message will not be caught by
            # the on_message handler)

            pong = NodeMessages.reply_create( {
                'researcher_id' : 'TEST_MESSAGING_RANDOM_ID_6735424425_DO_NOT_USE_ELSEWHERE',
                'node_id'       : 'node_1234',
                'success'       : True,
                'sequence'      : 12345,
                'command'       :'pong' } ).get_dict()

            self._m.default_send_topic = "general/researcher"

            self._m.send_message(pong)
            self.assertTrue( True, "fake pong message correctly sent")

        except:
            self.assertTrue( False, "fake pong message not sent")

        # give time to the on_message() handler to get and test the content of msg
        time.sleep(1.0)


class TestMessagingNode(unittest.TestCase):
    '''
    Test the Messaging class from the researcher point of view
    '''

    # once in test lifetime
    @classmethod
    def setUpClass(cls):

        # verify that a broker is available
        try:
            print("connecting to:", environ['MQTT_BROKER'], "/", environ['MQTT_BROKER_PORT'])
            cls._m = Messaging(None,
                               ComponentType.NODE,
                               'node_1234',
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


    # tests
    def test_messaging_researcher_00_init(self):

        self.assertEqual( self._m.default_send_topic, "general/researcher")
        self.assertEqual( self._m.on_message_handler, None)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
