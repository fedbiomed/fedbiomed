import queue
import socket
import threading


def start_udp_capture_server(host="127.0.0.1"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, 50064))
    port = sock.getsockname()[1]

    received = queue.Queue()

    def run():
        try:
            while True:
                data, _addr = sock.recvfrom(65535)
                received.put(data)
        except KeyboardInterrupt:
            print("\nStopping receiving...")
        finally:
            sock.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, received, thread


if __name__ == "__main__":
    port, received, thread = start_udp_capture_server()
    print(f"UDP capture server started on port {port}")
    while True:
        try:
            data = received.get(timeout=1)
            print(f"Received: {data.decode()}")
        except queue.Empty:
            pass
        except KeyboardInterrupt:
            print("\nStopping UDP syslog server.")
            break
