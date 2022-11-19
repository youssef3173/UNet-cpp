# This Bash Script Designed to Build and Run the Project: to run it use 'bash shell.sh' or 'sh shell.sh'
echo "----------------------------------------------------------"
echo "This Bash Script is Designed to Build and Run the Project:"
echo "----------------------------------------------------------"
echo "                                                          "

if [ ! -d /mnt/c/Users/Youssef/Desktop/CPP/AlexNet-cpp/build/ ]; then
	mkdir -p build && cd build/;
	cmake ..;
else
	cd build/;
fi

make

