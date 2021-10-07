#
# inspired by https://github.com/eclipse/paho.mqtt.python
#
import socket

class FakeBroker:
    def __init__(self, port = 9999):
        # Bind to "localhost" for maximum performance, as described in:
        # http://docs.python.org/howto/sockets.html#ipc
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(30)

        if port < 1024:
            port = 9999
        self._port = port

        sock.bind(("localhost", port))
        sock.listen(1)

        self._sock = sock
        self._conn = None

    def start(self):
        if self._sock is None:
            raise ValueError('Socket is not open')

        (conn, address) = self._sock.accept()
        conn.settimeout(10)
        self._conn = conn

    def finish(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def receive_packet(self, num_bytes):
        if self._conn is None:
            raise ValueError('Connection is not open')

        packet_in = self._conn.recv(num_bytes)
        print("RECV:", packet_in)
        return packet_in

    def send_packet(self, packet_out):
        if self._conn is None:
            raise ValueError('Connection is not open')

        count = self._conn.send(packet_out)
        print("SEND:", packet_out)
        return count
