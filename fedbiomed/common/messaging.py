'''
wrapper around the mqtt broket

This allows to replace mqqt library without changing the API of Messaging
'''


import socket
from typing import Any, Callable, Union

import paho.mqtt.client as mqtt

from fedbiomed.common import json
from fedbiomed.common.constants  import ComponentType, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedMessagingError
import fedbiomed.common.message as message
from fedbiomed.common.logger import logger


class Messaging:
    """ This class represents the messager,
    (MQTT messaging facility)."""

    def __init__(self,
                 on_message: Callable[[dict], None],
                 messaging_type: ComponentType,
                 messaging_id: Union[int, str],
                 mqtt_broker: str = 'localhost',
                 mqtt_broker_port: int = 1883):
        """ Constructor of the messaging class.
        Creates an instance of MQTT Client, and MQTT message handler.
        Creates topics on which to send messages through Messager.
        Topics in MQTT work as a channel allowing to filter shared information
        between connected clients

        Args:
            on_message (Callable): function that should be executed when
            a message is received
            messaging_type (ComponentType): describes incoming message sender.
            1 for researcher, 2 for node
            messaging_id ([int]): messaging id
            mqtt_broker (str, optional): IP address / URL. Defaults to
            "localhost".
            mqtt_broker_port (int, optional): Defaults to 80 (http
            default port).
        """
        self._messaging_type = messaging_type
        self._messaging_id = str(messaging_id)
        self._is_connected = False
        self._is_failed = False

        # Client() will generate a random client_id if not given
        # this means we choose not to use the {node,researcher}_id for this purpose
        self._mqtt = mqtt.Client()
        # defining a client.
        # defining MQTT 's `on_connect` and `on_message` handlers
        # (see MQTT paho documentation for further information
        # _ https://github.com/eclipse/paho.mqtt.python)
        self._mqtt.on_connect = self.on_connect
        self._mqtt.on_message = self.on_message
        self._mqtt.on_disconnect = self.on_disconnect

        self._mqtt_broker = mqtt_broker
        self._mqtt_broker_port = mqtt_broker_port

        # memorize mqqt parameters
        self._broker_host = mqtt_broker
        self._broker_port = mqtt_broker_port

        # protection for logger initialisation (mqqt handler)
        self._logger_handler_installed = False

        self._on_message_handler = on_message  # store the caller's mesg handler
        if on_message is None:
            logger.warning("no message handler defined")

        if self._messaging_type is ComponentType.RESEARCHER:
            self._default_send_topic = 'general/nodes'
        elif self._messaging_type is ComponentType.NODE:
            self._default_send_topic = 'general/researcher'
        else:
            self._default_send_topic = None


    def on_message(self,
                   client: mqtt.Client,
                   userdata: Any,
                   msg: Union[str, bytes]):
        """callback called when a new MQTT message is received
        the msg is processed and forwarded to the node/researcher
        to be treated/stored/whatever

        Args:
            client (mqtt.Client): mqtt on_message arg, client instance (unused)
            userdata (Any): mqtt on_message arg (unused)
            msg: mqtt on_message arg
        """

        if self._on_message_handler is not None:
            message = json.deserialize_msg(msg.payload)
            self._on_message_handler( msg = message, topic = msg.topic)
        else:
            logger.warning("no message handler defined")

    def on_connect(self,
                   client: mqtt.Client,
                   userdata: Any,
                   flags: dict,
                   rc: int):
        """callback for when the client receives a CONNACK response from the server.

        Args:
            client (mqtt.Client): mqtt on_message arg (unused)
            userdata: mqtt on_message arg, private user data (unused)
            flags (dict): mqtt on_message arg, response flag sent by the
            broker (unused)
            rc (int): mqtt on_message arg, connection result
        """

        if rc == 0:
            logger.info("Messaging " + str(self._messaging_id) +
                        " successfully connected to the message broker, object = " + str(self))
        else:
            msg = ErrorNumbers.FB101.value + ": " + str(self._messaging_id) + " could not connect to the message broker"
            logger.delMqttHandler()  # just in case !
            self._logger_handler_installed = False

            logger.critical(msg)
            self._is_failed = True
            raise FedbiomedMessagingError(msg)

        if self._messaging_type is ComponentType.RESEARCHER:
            for channel in ('general/researcher', 'general/monitoring'):
                result, _ = self._mqtt.subscribe(channel)
                if result != mqtt.MQTT_ERR_SUCCESS:
                    logger.error("Messaging " + str(self._messaging_id) + "failed subscribe to channel" + str(channel))
                    self._is_failed = True

            # PoC subscribe also to error channel
            result, _ = self._mqtt.subscribe('general/logger')
            if result != mqtt.MQTT_ERR_SUCCESS:
                logger.error("Messaging " + str(self._messaging_id) + "failed subscribe to channel general/error")
                self._is_failed = True

        elif self._messaging_type is ComponentType.NODE:
            for channel in ('general/nodes', 'general/' + self._messaging_id):
                result, _ = self._mqtt.subscribe(channel)
                if result != mqtt.MQTT_ERR_SUCCESS:
                    logger.error("Messaging " + str(self._messaging_id) + " failed subscribe to channel" + str(channel))
                    self._is_failed = True

            if not self._logger_handler_installed:
                # add the MQTT handler for logger
                # this should be done once.
                # This is sldo tested by the addHandler() method, but
                # it may raise a MQTT message (that we prefer not to send)
                logger.addMqttHandler(
                    mqtt          = self._mqtt,
                    node_id       = self._messaging_id
                )
                # to get Train/Epoch messages on console and on MQTT
                logger.setLevel("DEBUG")
                self._logger_handler_installed = True

        self._is_connected = True

    def on_disconnect(self, client, userdata, rc):
        self._is_connected = False

        if rc == 0:
            # should this ever happen ? we're not disconnecting intentionally yet
            logger.info("Messaging " + str(self._messaging_id) + " disconnected without error")
        else:
            # see MQTT specs : when another client connects with same client_id,
            # the previous one is disconnected
            # https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html#_Toc3901205
            logger.error("Messaging " + str(self._messaging_id) + " disconnected with error code rc = " + str(rc) +
                         " - Hint: check for another instance of the same component running or for communication error")

            self._is_failed = True
            # quit messaging to avoid connect/disconnect storm in case multiple nodes with same id
            raise SystemExit

    def start(self, block=False):
        """ this method calls the loop function of mqtt.
        Starts message handling by the library.

        Args:
            block (bool, optional): if True: calls the loop_forever method in
                MQTT (blocking loop)
                else, calls the loop_start method
                (non blocking loop).
                `loop_start` calls a background thread
                for messaging.
                See Paho MQTT documentation
                (https://github.com/eclipse/paho.mqtt.python)
                for further information. Defaults to False.
        """
        # will try a connect even if is_failed or is_connected, to give a chance to resolve problems

        try:
            self._mqtt.connect(self._mqtt_broker, self._mqtt_broker_port, keepalive=60)
        except (ConnectionRefusedError, TimeoutError, socket.timeout ) as e:

            logger.delMqttHandler()  # just in case !
            self._logger_handler_installed = False

            msg = ErrorNumbers.FB101.value + "(error = " + str(e) + ")"
            logger.critical(msg)
            raise FedbiomedMessagingError(msg)

        if block:
            # TODO : not used, should probably be removed
            self._mqtt.loop_forever()
        elif not self._is_connected:
            self._mqtt.loop_start()
            while not self._is_connected:
                pass

    def stop(self):
        """
        This method stops the loop started using `loop_start` method -
        ie the non-blocking loop triggered with `Messaging.start(block=True)`
        only. It stops the background thread for messaging.
        """
        # will try a stop even if is_failed or not is_connected, to give a chance to clean state
        self._mqtt.loop_stop()

    def send_message(self, msg: dict, client: str = None):
        """This method sends a message to a given client

        Args:
            msg (dict): the content of a message
            client ([str], optional): defines the channel to which the
                                message will be sent. Defaults to None(all
                                clients)
        """
        if self._is_failed:
            logger.error('Messaging has failed, will not try to send message')
            return
        elif not self._is_connected:
            logger.error('Messaging is not connected, will not try to send message')
            return

        if client is None:
            channel = self._default_send_topic
        else:
            channel = "general/" + str(client)
        if channel is not None:
            messinfo = self._mqtt.publish(channel, json.serialize_msg(msg))
            if messinfo.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error("Messaging " +
                             str(self._messaging_id) +
                             " failed sending message with code rc = ",
                             str(messinfo.rc) +
                             " object = " +
                             str(self) +
                             " message = " +
                             str(msg))
                self._is_failed = True
        else:
            logger.warning("send_message: channel must be specific (None at the moment)")

    def send_error(self, errnum: ErrorNumbers, extra_msg: str = "", researcher_id: str = "<unknown>"):
        """
        node sends error through mqtt

        remark: difference with send_message() is that we do extra tests
        before sending the message
        """

        if self._messaging_type != ComponentType.NODE:
            logger.warning("this component (" +
                           self._messaging_type +
                           ") cannot send error message (" +
                           errnum.value +
                           ") through MQTT")
            return

        if not self._is_connected:
            logger.delMqttHandler()  # just in case
            self._logger_handler_installed = False

            msg = "MQTT not initialized yet (error to transmit=" + errnum.value + ")"
            logger.critical(msg)
            raise FedbiomedMessagingError(msg)


        # format error message and send it
        msg = dict(
            command       = 'error',
            errnum        = errnum,
            node_id       = self._messaging_id,
            extra_msg     = extra_msg,
            researcher_id = researcher_id
        )

        # just check the syntax bfore sendind
        _ = message.NodeMessages.reply_create(msg)
        self._mqtt.publish("general/researcher", json.serialize_msg(msg))


    def is_failed(self):
        '''
        getter for the is_failed status flag
        '''
        return self._is_failed

    def is_connected(self):
        '''
        getter for the is_connected status flag
        '''
        return self._is_connected

    def default_send_topic(self):
        '''
        getter for default_send_topic
        '''
        return self._default_send_topic
