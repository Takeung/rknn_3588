docker run -it --name rknn_starter_rk3588 \
               --privileged   --ipc=host  \
               -v /dev/bus/usb:/dev/bus/usb  \
               -v /tmp/.X11-unix:/tmp/.X11-unix  \
               -p 8090:22  -e DISPLAY=:0  --network host  \
               -v /data/code/chendeqiang/rknn:/home/rknn  \
               bboyhanat/ubuntu2004-ros2foxy-rknn150:v1.3.1 

