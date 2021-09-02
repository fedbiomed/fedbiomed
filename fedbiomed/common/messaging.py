from enum import Enum
from typing import Any, Callable, Union

import paho.mqtt.client as mqtt

from fedbiomed.common import json


class MessagingType(Enum):
    """Enumeration class, used to charactierize
    sender of a message (whether it is coming from 
    a researcher instance or a node instance)

    """
    RESEARCHER = 1
    NODE = 2

class Messaging:
    """ This class represents the Messager,
    (MQTT messaging facility)."""

    def __init__(self,
                 on_message: Callable[[dict], None],
                 messaging_type: MessagingType,
                 messaging_id: int, 
                 mqtt_broker:str='localhost',
                 mqtt_broker_port:int=80):
        """ Constructor of the messaging class.
        Creates an instance of MQTT Client, and MQTT message handler.
        Creates topics on which to send messages through Messager. 
        Topics in MQTT work as a channel allowing to filter shared information
        between connected clients
        
        Args:
            on_message (Callable): function that should be executed when a message is received
            messaging_type (MessagingType): describes incoming message sender.
            1 for researcher, 2 for node
            messaging_id ([int]): messaging id
            mqtt_broker (str, optional): IP address / URL. Defaults to "localhost".
            mqtt_broker_port (int, optional): Defaults to 80 (http default port).
        """        
        self.messaging_type = messaging_type
        self.messaging_id = str(messaging_id)
        self.mqtt = mqtt.Client(client_id=messaging_id) # defining a client.
        # defining MQTT 's `on_connect` and `on_message` handlers
        # (see MQTT paho documentation for further information
        # _ https://github.com/eclipse/paho.mqtt.python)
        self.mqtt.on_connect = self.on_connect
        self.mqtt.on_message = self.on_message  # refering to the method: not to `on_message` handler
        self.mqtt.connect(mqtt_broker, mqtt_broker_port, keepalive=60)

        self.on_message_handler = on_message # store the caller's message handler

        if self.messaging_type is MessagingType.RESEARCHER:
            self.default_send_topic = 'general/clients'
        elif self.messaging_type is MessagingType.NODE:
            self.default_send_topic = 'general/server'
        else: # should not occur
            self.default_send_topic = None

        self.is_connected = False

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
        message = json.deserialize_msg(msg.payload)
        self.on_message_handler(message)


    def on_connect(self, client: mqtt.Client, userdata: Any, flags: dict, rc: int):
        """callback for when the client receives a CONNACK response from the server.

        Args:
            client (mqtt.Client): mqtt on_message arg (unused)
            userdata: mqtt on_message arg, private user data (unused)
            flags (dict): mqtt on_message arg, response flag sent by the broker (unused)
            rc (int): mqtt on_message arg, connection result 
        """        
        print("Messaging " + self.messaging_id + " connected with result code " + str(rc))
        if self.messaging_type is MessagingType.RESEARCHER:
            self.mqtt.subscribe('general/server')
        elif self.messaging_type is MessagingType.NODE:
            self.mqtt.subscribe('general/clients')
            self.mqtt.subscribe('general/' + self.messaging_id)

        self.is_connected = True

    def start(self, block=False):
        """ this method calls the loop function of mqtt. Maintains network
        traffic flows through the broker.

        Args:
            block (bool, optional): if True: calls the loop_forever method in MQTT 
                                    (blocking loop)
                                    else, calls the loop_start method (non blocking loop).
                                    `loop_start` calls a background thread for messaging.
                                    See Paho MQTT documentation 
                                    (https://github.com/eclipse/paho.mqtt.python)
                                    for further information. Defaults to False.
        """        
        if block:
            self.mqtt.loop_forever()
        elif not self.is_connected:
            self.mqtt.loop_start()
            while not self.is_connected:
                pass

    def stop(self):
        """
        this method stops the loop started using `loop_start` method -
        ie the non-blocking loop triggered with `Messaging.start(block=True)`
        only. It stops the background thread for messaging.
        """        
        self.mqtt.loop_stop()

    def send_message(self, msg: dict, client: str=None):
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
            print("send_message: channel must be specific (None at the moment)")
