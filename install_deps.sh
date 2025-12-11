set -e

echo "Installing numpy and Cython first..."
pip install --no-cache-dir numpy>=1.24.0 Cython>=0.29.0

echo "Installing pkuseg..."
pip install --no-cache-dir --no-build-isolation pkuseg==0.0.25

echo "Installing remaining dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "All dependencies installed successfully!"