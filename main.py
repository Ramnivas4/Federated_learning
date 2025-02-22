import subprocess
import time
import platform

def start_federated_learning():
    system_platform = platform.system()

    if system_platform == "Windows":
        # Windows: Open new Command Prompt windows for server and client
        server_cmd = 'start cmd /k "venv\\Scripts\\activate && python server.py"'
        client_cmd = 'start cmd /k "venv\\Scripts\\activate && python client.py"'
    else:
        # Linux/macOS: Open new terminal windows and keep them open
        server_cmd = 'gnome-terminal -- bash -c "source venv/bin/activate && python server.py; exec bash"'
        client_cmd = 'gnome-terminal -- bash -c "source venv/bin/activate && python client.py; exec bash"'

    # Start the server in a new terminal
    subprocess.Popen(server_cmd, shell=True)
    print("Server started in a new terminal...")

    # Wait for 5 seconds
    time.sleep(5)

    # Start the client in another new terminal
    subprocess.Popen(client_cmd, shell=True)
    print("Client started in a new terminal...")

if __name__ == "__main__":
    start_federated_learning()
