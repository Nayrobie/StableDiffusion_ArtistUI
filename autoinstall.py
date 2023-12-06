import subprocess
import sys
import os
import platform

def create_virtualenv(path, python_executable):
    if not os.path.exists(path):
        os.makedirs(path)
    subprocess.check_call([python_executable, '-m', 'venv', path])

def install_requirements(venv_path, requirements_file):
    # Utilisation de python -m pip pour garantir l'utilisation du pip correct dans l'environnement virtuel
    python_executable = os.path.join(venv_path, 'Scripts', 'python' if platform.system() == 'Windows' else 'bin/python')
    subprocess.check_call([python_executable, '-m', 'pip', 'install', '-r', requirements_file])

python_executable = sys.executable
venv_path = 'genai_env'
requirements_file = 'requirements.txt'

try:
    create_virtualenv(venv_path, python_executable)
    install_requirements(venv_path, requirements_file)
except subprocess.CalledProcessError as e:
    print(f"Une erreur est survenue lors de l'exécution du script : {e}")
    print(f"Code de sortie : {e.returncode}")
    print(f"Commande exécutée : {' '.join(e.cmd)}")
