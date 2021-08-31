import uuid
from enum import Enum
import paho.mqtt.client as mqtt

from fedbiomed.common import json


class MessagingType(Enum):
    RESEARCHER = 1
    NODE = 2


class Messaging:
    """ This class represents the MQTT messaging."""

    def __init__(self, on_message, messaging_type: MessagingType, messaging_id,
                 mqtt_broker='localhost', mqtt_broker_port=80):
        """ Constructor of the messaging class

        Args:
            on_message ([function]): function that should be executed when a message is received
            messaging_type (MessagingType): 1 for researcher, 2 for researcher
            messaging_id ([int]): messaging id
            mqtt_broker_port (int, optional): Defaults to 80.
        """
        self.messaging_type = messaging_type
        self.messaging_id = str(uuid.uuid4()) if messaging_type == MessagingType.RESEARCHER else str(messaging_id)
        self.mqtt = mqtt.Client(client_id=self.messaging_id)
        self.mqtt.on_connect = self.on_connect
        self.mqtt.on_message = self.on_message
        self.mqtt.connect(mqtt_broker, mqtt_broker_port, keepalive=60)

        self.on_message_handler = on_message  # store the caller's mesg handler

        if self.messaging_type is MessagingType.RESEARCHER:
            self.default_send_topic = 'general/clients'
        elif self.messaging_type is MessagingType.NODE:
            self.default_send_topic = 'general/server'
        else:  # should not occur
            self.default_send_topic = None

        self.is_connected = False

    def on_message(self, client, userdata, msg):
        """called then a new MQTT message is received
        the msg is processes and forwarded to the node/researcher
        to be treated/stored/whatever

        Args:
            client: mqtt on_message arg
            userdata: mqtt on_message arg
            msg: mqtt on_message arg
        """
        message = json.deserialize_msg(msg.payload)
        self.on_message_handler(message)

    def on_connect(self, client, userdata, flags, rc):
        """[summary]

        Args:
            client: mqtt on_message arg
            userdata: mqtt on_message arg
            flags: mqtt on_message arg
            rc: mqtt on_message arg
        """
        print("Messaging " + self.messaging_id + " connected with result code " + str(rc))
        if self.messaging_type is MessagingType.RESEARCHER:
            self.mqtt.subscribe('general/server')
        elif self.messaging_type is MessagingType.NODE:
            self.mqtt.subscribe('general/clients')
            self.mqtt.subscribe('general/' + self.messaging_id)

        self.is_connected = True

    def start(self, block=False):
        """ this method calls the loop function of mqtt

        Args:
            block (bool, optional): if True: calls the loop_forever method 
                                    else, calls the loop_start method
        """
        if block:
            self.mqtt.loop_forever()
        elif not self.is_connected:
            self.mqtt.loop_start()
            while not self.is_connected:
                pass

    def stop(self):
        """
        this method stops the loop 
        """
        self.mqtt.loop_stop()

    def send_message(self, msg: dict, client=None):
        """This method sends a message to a given client

        Args:
            msg (dict): the content of a message
            client ([str], optional): defines the channel to which the 
                                message will be sent. Defaults to None(all clients)
        """
        if client is None:
            channel = self.default_send_topic
        else:
            channel = "general/" + client

        if channel is not None:
            self.mqtt.publish(channel, json.serialize_msg(msg))
        else:
            print("send_message: channel must ne specifiec (None at the moment)")
