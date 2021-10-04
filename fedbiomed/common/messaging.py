from enum import Enum
from typing import Any, Callable, Union

import paho.mqtt.client as mqtt

from fedbiomed.common import json
from fedbiomed.common.logger import logger
from fedbiomed.common.logger import DEFAULT_LOG_TOPIC


class MessagingType(Enum):
    """Enumeration class, used to characterize
    context of message handling (whether it is done in
    a researcher instance or a node instance)

    """
    RESEARCHER = 1
    NODE = 2


class Messaging:
    """ This class represents the messager,
    (MQTT messaging facility)."""

    def __init__(self,
                 on_message: Callable[[dict], None],
                 messaging_type: MessagingType,
                 messaging_id: int,
                 mqtt_broker: str = 'localhost',
                 mqtt_broker_port: int = 80):
        """ Constructor of the messaging class.
        Creates an instance of MQTT Client, and MQTT message handler.
        Creates topics on which to send messages through Messager.
        Topics in MQTT work as a channel allowing to filter shared information
        between connected clients

        Args:
            on_message (Callable): function that should be executed when
            a message is received
            messaging_type (MessagingType): describes incoming message sender.
            1 for researcher, 2 for node
            messaging_id ([int]): messaging id
            mqtt_broker (str, optional): IP address / URL. Defaults to
            "localhost".
            mqtt_broker_port (int, optional): Defaults to 80 (http
            default port).
        """
        self.messaging_type = messaging_type
        self.messaging_id = str(messaging_id)
        self.is_connected = False
        self.is_failed = False

        # Client() will generate a random client_id if not given
        # this means we choose not to use the {node,researcher}_id for this purpose
        self.mqtt = mqtt.Client()
        # defining a client.
        # defining MQTT 's `on_connect` and `on_message` handlers
        # (see MQTT paho documentation for further information
        # _ https://github.com/eclipse/paho.mqtt.python)
        self.mqtt.on_connect = self.on_connect
        self.mqtt.on_message = self.on_message
        self.mqtt.on_disconnect = self.on_disconnect

        self.mqtt_broker = mqtt_broker
        self.mqtt_broker_port = mqtt_broker_port

        # memorize mqqt parameters
        self._broker_host = mqtt_broker
        self._broker_port = mqtt_broker_port

        # protection for logger initialisation (mqqt handler)
        self.logger_initialized = False

        self.on_message_handler = on_message  # store the caller's mesg handler

        if self.messaging_type is MessagingType.RESEARCHER:
            self.default_send_topic = 'general/clients'
        elif self.messaging_type is MessagingType.NODE:
            self.default_send_topic = 'general/server'
        else:  # should not occur
            self.default_send_topic = None


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

        # did not decide how to manage general/logger messages yet
        print("#### msg received - topic =", str(msg.topic), " content =", str(json.deserialize_msg(msg.payload)))
        if str(msg.topic) == "general/logger":
            return

        message = json.deserialize_msg(msg.payload)
        self.on_message_handler(message)

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
            logger.info("Messaging " + str(self.messaging_id) + " successfully connected to the message broker, object = " + str(self))
        else:
            logger.error("Messaging " + str(self.messaging_id) + " could not connect to the message broker, object = " + str(self))
            self.is_failed = True

        if self.messaging_type is MessagingType.RESEARCHER:
            result, _ = self.mqtt.subscribe('general/server')
            if result != mqtt.MQTT_ERR_SUCCESS:
                logger.error("Messaging " + str(self.messaging_id) + "failed subscribe to channel general/server")
                self.is_failed = True

            # PoC subscibe also to error channel
            result, _ = self.mqtt.subscribe('general/logger')
            if result != mqtt.MQTT_ERR_SUCCESS:
                logger.error("Messaging " + str(self.messaging_id) + "failed subscribe to channel general/error")
                self.is_failed = True
        elif self.messaging_type is MessagingType.NODE:
            for channel in ('general/clients', 'general/' + self.messaging_id):
                result, _ = self.mqtt.subscribe(channel)
                if result != mqtt.MQTT_ERR_SUCCESS:
                    logger.error("Messaging " + str(self.messaging_id) + " failed subscribe to channel" + str(channel))
                    self.is_failed = True

            if not self.logger_initialized:
                # add the MQTT handler for logger
                # this should be done once.
                # This is sldo tested by the addHandler() method, but
                # it may raise a MQTT message (that we prefer not to send)
                logger.addMqttHandler(
                    mqtt          = self.mqtt,
                    client_id     = self.messaging_id
                )
                # to get Train/Epoch messages on console and on MQTT
                logger.setLevel("DEBUG")
                self.logger_initialized = True


        self.is_connected = True

    def on_disconnect(self, client, userdata, rc):
        self.is_connected = False

        if rc == 0:
            # should this ever happen ? we're not disconnecting intentionally yet
            logger.info("Messaging " + str(self.messaging_id) + " disconnected without error, object = " + str(self))
        else:
            # see MQTT specs : when another client connects with same client_id, the previous one
            # is disconnected https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html#_Toc3901205

            #print("[ERROR] Messaging ", self.messaging_id, " disconnected with error code rc = ", rc, " object = ", self,
            #    " - Hint: check for another instance of the same component running or for communication error")
            logger.error("Messaging " + str(self.messaging_id) + " disconnected with error code rc = " + str(rc) +
                         " object = " + str(self) +
                         " - Hint: check for another instance of the same component running or for communication error")

            self.is_failed = True
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

        self.mqtt.connect(self.mqtt_broker, self.mqtt_broker_port, keepalive=60)
        if block:
            # TODO : not used, should probably be removed
            self.mqtt.loop_forever()
        elif not self.is_connected:
            self.mqtt.loop_start()
            while not self.is_connected:
                pass

    def stop(self):
        """
        This method stops the loop started using `loop_start` method -
        ie the non-blocking loop triggered with `Messaging.start(block=True)`
        only. It stops the background thread for messaging.
        """
        # will try a stop even if is_failed or not is_connected, to give a chance to clean state
        self.mqtt.loop_stop()

    def send_message(self, msg: dict, client: str = None):
        """This method sends a message to a given client

        Args:
            msg (dict): the content of a message
            client ([str], optional): defines the channel to which the
                                message will be sent. Defaults to None(all
                                clients)
        """
        if self.is_failed:
            logger.error('Messaging has failed, will not try to send message')
            return
        elif not self.is_connected:
            logger.error('Messaging is not connected, will not try to send message')
            return

        if client is None:
            channel = self.default_send_topic
        else:
            channel = "general/" + client
        if channel is not None:
            messinfo = self.mqtt.publish(channel, json.serialize_msg(msg))
            if messinfo.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error("Messaging " +
                             str(self.messaging_id) +
                             " failed sending message with code rc = ",
                             str(messinfo.rc) +
                             " object = " +
                             str(self) +
                             " message = " +
                             str(msg))
                self.is_failed = True
        else:
            logger.warning("send_message: channel must be specifiec (None at the moment)")
