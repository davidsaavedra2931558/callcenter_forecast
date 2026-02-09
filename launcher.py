import subprocess
import webbrowser
import time
import os
import sys

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)

dashboard_path = resource_path("dashboard.py")

cmd = f'"{sys.executable}" -m streamlit run "{dashboard_path}" --server.headless=true'

process = subprocess.Popen(cmd, shell=True)

time.sleep(6)
webbrowser.open("http://localhost:8501")

process.wait()
