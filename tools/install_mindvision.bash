#!/bin/bash
echo -e "\n\n>>> Setup packages" && sleep 1
sudo apt-get update
#sudo apt install ros-jazzy-serial-driver ros-jazzy-asio-cmake-module -y

echo -e "\n\n>>> MindVision SDK" && sleep 1
if [ -d /usr/include/mindvision/ ]; then
    echo "mindvision-sdk already installed"
else
    echo ">>> start install mindvision-sdk"
    mkdir mindvision-sdk
    wget https://www.mindvision.com.cn/wp-content/uploads/2023/08/linuxSDK_V2.1.0.37.tar.gz -O mindvision-sdk/SDK.tar.gz
    tar -zxvf mindvision-sdk/SDK.tar.gz --directory=mindvision-sdk
    sed -i 's/usr\/include/usr\/include\/mindvision/g' mindvision-sdk/install.sh
    sed -i 17a\\"mkdir -p /usr/include/mindvision" mindvision-sdk/install.sh
    cd mindvision-sdk && sudo bash ./install.sh && cd ..
    echo ">>> successfully install mindvision-sdk"
    echo ">>> 不需要重启电脑！！！！！"
    echo ">>> 说什么都不需要重启电脑！！！！！"
    rm -r mindvision-sdk
fi
