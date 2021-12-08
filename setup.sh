python3 -m venv supermariobros-env
source supermariobros-env/bin/activate

# Installation Reqs
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
pip install -r requirements.txt
pip install -r test-requirements.txt