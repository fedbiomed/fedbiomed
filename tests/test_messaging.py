import unittest
from unittest.mock import patch, PropertyMock, Mock
import time

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase, NodeTestCase
#############################################################

from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedMessagingError
from fedbiomed.common.message import NodeMessages
from fedbiomed.common.messaging import Messaging

from fedbiomed.researcher.environ import environ


class TestMessaging(unittest.TestCase):
    '''
    Test the Messaging class connect/disconnect
    '''

    def setUp(self):
        '''
        before all the tests
        '''
        self._m = Messaging(on_message=None,
                            messaging_type=None,
                            messaging_id=1234,
                            mqtt_broker="1.2.3.4",
                            mqtt_broker_port=1)

        pass

    def test_messaging_00_bad_init(self):
        '''
        bad __init__ calls
        '''

        self.assertFalse(self._m.is_connected())
        self.assertEqual(self._m.default_send_topic(), None)

        try:
            self._m.start()
            self.assertFalse(True, "Should not connect to fake server (KO)")
        except:
            self.assertTrue(True, "Should not connect to fake server (OK)")

    def test_messaging_01_researcher_init(self):
        '''
        as test name says...
        '''
        self._m = Messaging(on_message=None,
                            messaging_type=ComponentType.RESEARCHER,
                            messaging_id=1234,
                            mqtt_broker="1.2.3.4",
                            mqtt_broker_port=1)

        self.assertFalse(self._m.is_connected())
        self.assertEqual(self._m.default_send_topic(), "general/nodes")

    def test_messaging_02_node_init(self):
        '''
        as test name says...
        '''

        self._m = Messaging(on_message=None,
                            messaging_type=ComponentType.NODE,
                            messaging_id=1234,
                            mqtt_broker="1.2.3.4",
                            mqtt_broker_port=1)

        self.assertFalse(self._m.is_connected())
        self.assertEqual(self._m.default_send_topic(), "general/researcher")

    def test_messaging_03_connect(self):
        '''
        as test name says...
        '''

        self._m.on_connect(None,  # client
                           None,  # userdata
                           None,  # flags
                           0  # rc (ok)
                           )
        self.assertTrue(self._m.is_connected())

        try:
            self._m.on_connect(None,  # client
                               None,  # userdata
                               None,  # flags
                               1  # rc (error)
                               )
            self.assertFalse(True, "bad disconnexion")
        except:
            self.assertTrue(True, "bad disconnexion")
            self.assertTrue(self._m.is_failed())

    def test_messaging_04_disconnect(self):
        '''
        as test name says...
        '''

        # disconnexion from server

        # ugly but we dont have/want a setter for this
        self._m._is_connected = True

        self._m.on_disconnect(None,  # client
                              None,  # userdata
                              0)  # rc (OK)
        self.assertFalse(self._m.is_connected())

        # failed disconnexion from server
        self._m._is_connected = True
        try:
            self._m.on_disconnect(None,  # client
                                  None,  # userdata
                                  1)  # rc (error)
            self.assertFalse(True, "Disconnexion should failed and raise SystemExit")
        except SystemExit:
            self.assertTrue(True, "Disconnexion has failed and raised SystemExit")
        except:
            self.assertFalse(True, "Disconnexion has failed and dit not raised SystemExit")

        self.assertFalse(self._m.is_connected())
        self.assertTrue(self._m.is_failed())

    def test_messaging_05_bad_start(self):
        '''
        as test name says...
        '''

        with patch('paho.mqtt.client.Client.connect',
                   new_callable=PropertyMock, side_effect=ConnectionRefusedError('Boom!')):
            try:
                self._m.start()
                self.assertFalse(True, "Connexion exception not detected at start()")
            except FedbiomedMessagingError:
                self.assertTrue(True, "Connexion exception detected at start()")
            except Exception:
                self.assertFalse(True, "Bad Exception for connexion exception at start()")

    @patch('paho.mqtt.client.Client.loop_forever', Mock(return_value=True))
    @patch('paho.mqtt.client.Client.connect', Mock(return_value=True))
    def test_messaging_06_good_start(self):
        '''
        as test name says...
        '''

        try:
            self._m.start(block=True)
            self.assertTrue(True, "Connexion correctely started")
        except Exception:
            self.assertFalse(True, "Connexion correctly started and detected as no")

        self._m.stop()


class TestMessagingResearcher(ResearcherTestCase):
    '''
    Test the Messaging class from the researcher point of view
    '''

    @classmethod
    def setUpClass(cls):
        '''
        connect to the broker and setup a global variable
        used to skip the test is no broker is present
        '''

        super().setUpClass()

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
        except Exception:
            cls._broker_ok = False

        time.sleep(1.0)  # give some time to the thread to listen to the server
        print("MQTT connexion status =", cls._broker_ok)

        pass

    @classmethod
    def tearDownClass(cls):
        '''
        disconnect to the broker if broker was present at init time
        '''
        super().setUpClass()

        if not cls._broker_ok:
            cls._m.stop()
        pass

    def setUp(self):
        '''
        before all the tests
        '''
        pass

    def tearDown(self):
        '''
        after all the tests
        '''
        pass

    @classmethod
    def on_message(cls, msg, topic):
        '''
        on_message MQTT handler

        classmethod necessary to have access to self via cls
        '''

        print("RESH_RECV:", topic, msg)

        # verify the channel
        cls.assertTrue(cls, topic, "general/researcher")

        # verify the received msg only if sent by ourself
        if msg['researcher_id'] == 'TEST_MESSAGING_RANDOM_ID_6735424425_DO_NOT_USE_ELSEWHERE':
            cls.assertTrue(cls, msg['node_id'], "node_1234")
            cls.assertTrue(cls, msg['success'], True)
            cls.assertTrue(cls, msg['sequence'], 12345)
            cls.assertTrue(cls, msg['command'], 'pong')

    def test_messaging_researcher_00_init(self):
        '''
        test __init__()
        '''
        self.assertEqual(self._m.default_send_topic(), "general/nodes")

        # ugly, but we dont really hace/need a getter for this
        self.assertEqual(self._m._on_message_handler, TestMessagingResearcher.on_message)

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

            pong = NodeMessages.reply_create({
                'researcher_id': 'TEST_MESSAGING_RANDOM_ID_6735424425_DO_NOT_USE_ELSEWHERE',
                'node_id': 'node_1234',
                'success': True,
                'sequence': 12345,
                'command': 'pong'}).get_dict()

            # ugly, but we dont have/want a setter for this
            self._m._default_send_topic = "general/researcher"

            self._m.send_message(pong)
            self.assertTrue(True, "fake pong message correctly sent")

        except:
            self.assertTrue(False, "fake pong message not sent")

        # give time to the on_message() handler to get and test the content of msg
        time.sleep(1.0)


class TestMessagingNode(NodeTestCase):
    '''
    Test the Messaging class from the researcher point of view
    '''

    @classmethod
    def setUpClass(cls):
        '''
        connect to the broker and setup a global variable
        used to skip the test is no broker is present
        '''
        super().setUpClass()
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
        except Exception:
            cls._broker_ok = False

        time.sleep(1.0)  # give some time to the thread to listen to the server
        print("MQTT connexion status =", cls._broker_ok)

        pass

    @classmethod
    def tearDownClass(cls):
        '''
        disconnect to the broker if broker was present at init time
        '''
        super().setUpClass()

        if not cls._broker_ok:
            cls._m.stop()
        pass

    def setUp(self):
        '''
        before all the tests
        '''
        pass

    def tearDown(self):
        '''
        after all the tests
        '''
        pass

    def test_messaging_researcher_00_init(self):
        '''
        test __init__
        '''
        self.assertEqual(self._m.default_send_topic(), "general/researcher")

        # ugly, but we dont really hace/need a getter for this
        self.assertEqual(self._m._on_message_handler, None)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
