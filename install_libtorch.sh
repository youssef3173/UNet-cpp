NC="\033[0m"
GREEN="\033[0;32m"

echo -e " ${BLUE}------------- Install LibTorch for C++  -------------------${NC}"


wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip



echo -e " ${BLUE}------------  install some Python Libraries   ------------${NC}"

sudo apt install -y python3
apt install -y python3-pip

sudo apt-get install -y python3-matplotlib
sudo apt-get install -y python3-opencv
pip install -y tqdm

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu


