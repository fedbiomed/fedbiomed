from multiprocessing import Process
import socket
import sys
import os

from fedbiomed.researcher.environ import TENSORBOARD_RESULTS_DIR

class TensorboardSupervisor:
    def __init__(self, log_dp):
        self.server = TensorboardServer(log_dp)
        self.server.start()
        print("Started Tensorboard Server")
        self.chrome = ChromeProcess()
        print("Started Chrome Browser")
        self.chrome.start()

    def finalize(self):
        if self.server.is_alive():
            print('Killing Tensorboard Server')
            self.server.terminate()
            self.server.join()
        # As a preference, we leave chrome open -
        # but this may be amended similar to the method above


class TensorboardServer(Process):
    def __init__(self, log_dp):
        super().__init__()
        self.os_name = os.name
        self.log_dp = str(log_dp)
        self.daemon = False

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.log_dp}" 2> NUL')
        elif self.os_name == 'posix':  # Linux
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.log_dp}" '
                      f'--host {socket.gethostname()}')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')


class ChromeProcess(Process):
    def __init__(self):
        super().__init__()
        self.os_name = sys.platform
        self.daemon = False

    def run(self):
        url = f'http://{socket.gethostname()}:6006/'
        if 'windows' in self.os_name:  # Windows
            os.system(f'start chrome {url}')
        elif 'linux' in self.os_name:  # Linux
            os.system(f'google-chrome {url}')
        elif 'darwin' in self.os_name:
            os.system(f'open {url}')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')


def launch_tensorboard():
    tb_sup = None
    try:
        print('Starting tensorboard for client monitoring...')
        tb_sup = TensorboardSupervisor(TENSORBOARD_RESULTS_DIR)
    except KeyboardInterrupt:
        print('Stopping service...')
        tb_sup.finalize()
