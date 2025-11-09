
import sys
import socket
from streamlit.web import cli

def get_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

if __name__ == "__main__":
    port = get_free_port()
    sys.argv = ["streamlit", "run", "app.py", "--server.port",str(port)]
    cli.main()

