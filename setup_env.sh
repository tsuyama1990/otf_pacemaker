#!/bin/bash
set -e

echo "=============================================="
echo "Setting up environment for ace-active-carver"
echo "=============================================="

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install uv first."
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Initializing Python environment with uv..."
# Install dependencies defined in pyproject.toml
uv sync

echo "Creating directory structure..."
mkdir -p src tests data

echo ""
echo "=============================================="
echo "IMPORTANT: LAMMPS Installation Instructions"
echo "=============================================="
echo "This project requires a LAMMPS binary built with the PACE package."
echo "Automated compilation is disabled to prevent environment issues."
echo ""
echo "Please perform the following steps manually to build LAMMPS:"
echo ""
echo "1. Clone the LAMMPS repository:"
echo "   git clone -b stable https://github.com/lammps/lammps.git mylammps"
echo "   cd mylammps"
echo ""
echo "2. Create a build directory:"
echo "   mkdir build && cd build"
echo ""
echo "3. Configure with CMake (enabling Python and PACE package):"
echo "   cmake ../cmake \\"
echo "       -D PKG_PYTHON=ON \\"
echo "       -D PKG_ML-PACE=ON \\"
echo "       -D BUILD_SHARED_LIBS=ON \\"
echo "       -D PYTHON_EXECUTABLE=\$(uv python find)"
echo ""
echo "   # Note: You may need to add other packages as needed (e.g., -D PKG_MANYBODY=ON)"
echo ""
echo "4. Build and Install:"
echo "   make -j\$(nproc)"
echo "   make install"
echo ""
echo "   # Ensure the generated 'liblammps.so' is in your LD_LIBRARY_PATH"
echo "   # and the 'lammps' python module is accessible."
echo "   # Since we are using 'uv', the 'lammps' package in pyproject.toml"
echo "   # installs the python bindings, but they require the shared library."
echo ""
echo "=============================================="
echo "Setup complete (Python dependencies installed)."
echo "Please follow the instructions above for LAMMPS."
echo "=============================================="
