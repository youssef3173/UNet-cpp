NC="\033[0m"
BLUE="\033[0;32m"

echo -e " ${BLUE}----------- Install OpenCV for C++  -------------------${NC} "

mkdir -p install_opencv/ && cd install_opencv/

sudo apt-get update
sudo apt-get upgrade
sudo aot-get install libgtk2.0-dev pkg-config


sudo apt install -y g++
sudo apt-get install -y cmake make
sudo apt-get install -y git

echo ">> g++ version: "
g++ --version
echo ">> cmake version: "
cmake --version


git clone https://github.com/opencv.git
git clone https://github.com/opencv_contrib.git


mkdir -p build/ && cd build/

cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv
make -j4
sudo make install

cd ../    # return to install_opencv/
cd ../    # return to the main folder


# We do not need that aymore, because opencv are saved in '/usr/local/bin' and '/usr/local/lib':
rm -rf install_opencv
