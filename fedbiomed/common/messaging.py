import sys
from enum import Enum
import paho.mqtt.client as mqtt

from fedbiomed.common import json


class MessagingType(Enum):
    RESEARCHER = 1
    NODE = 2


class Messaging:
    """ This class represents the MQTT messaging."""

    def __init__(self, on_message, on_event, messaging_type: MessagingType, messaging_id,
                 mqtt_broker='localhost', mqtt_broker_port=80):
        """ Constructor of the messaging class

        Args:
            on_message (function(self, msg: dict)): function that is executed when a message is received
            on_event (function(self, event: dict)): function that is executed when an event occurs
            messaging_type (MessagingType): 1 for researcher, 2 for researcher
            messaging_id ([int]): messaging id
            mqtt_broker_port (int, optional): Defaults to 80.
        """
        self.messaging_type = messaging_type
        self.messaging_id = str(messaging_id)
        self.is_connected = False
        self.is_failed = False

        self.mqtt = mqtt.Client(client_id=self.messaging_id)
        self.mqtt.on_connect = self.on_connect
        self.mqtt.on_message = self.on_message
        self.mqtt.on_disconnect = self.on_disconnect

        self.mqtt_broker = mqtt_broker
        self.mqtt_broker_port = mqtt_broker_port

        self.on_message_handler = on_message  # store the caller's mesg handler
        self.on_event_handler = on_event # store the caller's event handler

        if self.messaging_type is MessagingType.RESEARCHER:
            self.default_send_topic = 'general/clients'
        elif self.messaging_type is MessagingType.NODE:
            self.default_send_topic = 'general/server'
        else:  # should not occur
            self.default_send_topic = None


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
        if rc == 0:
            print("[INFO] Messaging ", self.messaging_id, " successfully connected to the message broker, object = ", self)
        else:
            print("[ERROR] Messaging ", self.messaging_id, "could not connect to the message broker, object = ", self)
            sys.exit(-1)

        if self.messaging_type is MessagingType.RESEARCHER:
            result, _ = self.mqtt.subscribe('general/server')
            if result != mqtt.MQTT_ERR_SUCCESS:
                print("[ERROR] Messaging ", self.messaging_id, "failed subscribe to channel")
                sys.exit(-1)
        elif self.messaging_type is MessagingType.NODE:
            result, _ = self.mqtt.subscribe('general/clients')
            if result != mqtt.MQTT_ERR_SUCCESS:
                print("[ERROR] Messaging ", self.messaging_id, "failed subscribe to channel")
                sys.exit(-1)
            result, _ = self.mqtt.subscribe('general/' + self.messaging_id)
            if result != mqtt.MQTT_ERR_SUCCESS:
                print("[ERROR] Messaging ", self.messaging_id, "failed subscribe to channel")
                sys.exit(-1)
        self.is_connected = True

    def on_disconnect(self, client, userdata, rc):
        self.is_connected = False

        if rc == 0:
            # should this ever happen ? we're not disconnecting intentionally yet
            print("[INFO] Messaging ", self.messaging_id, " disconnected without error, object = ", self)
        else:
            # see MQTT specs : when another client connects with same client_id, the previous one
            # is disconnected https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html#_Toc3901205
            
            #print("[ERROR] Messaging ", self.messaging_id, " disconnected with error code rc = ", rc, " object = ", self,
            #    " - Hint: check for another instance of the same component running or for communication error")
            error_message = "[ERROR] Messaging " +  str(self.messaging_id) + " disconnected with error code rc = " + \
                str(rc) +  " object = " + str(self) + \
                " - Hint: check for another instance of the same component running or for communication error"
            print(error_message)
            #sys.exit(-1)
            #self.on_message_handler('ERROR', { 'message': error_message })
            self.on_event_handler({ 'type': 'ERROR_DISCONNECT', 'message': error_message })
            raise SystemExit

    def start(self, block=False):
        """ this method calls the loop function of mqtt

        Args:
            block (bool, optional): if True: calls the loop_forever method 
                                    else, calls the loop_start method
        """
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
            messinfo = self.mqtt.publish(channel, json.serialize_msg(msg))
            if messinfo.rc != mqtt.MQTT_ERR_SUCCESS:
                print("[ERROR] Messaging ", self.messaging_id, "failed sending message with code rc = ",
                messinfo.rc, " object = ", self, " message = ", msg)
                sys.exit(-1)
        else:
            print("[ERROR] Messaging ", self.messaging_id, " send_message: channel must be specified (None at the moment)")
            sys.exit(-1)
