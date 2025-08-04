# Create an activate a virtuell environment
python3 -m venv OmnisafeEnv \
source OmnisafeEnv/bin/activate

# Install dependencies in this order:
cd omnisafe \
pip install -e .

cd safety-gymnasium \
pip install -e .

# Start training
python3 omnisafe/train_velocities.py