正常工作的版本：cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv:v1.1
1. 容器的SPEC:
	* py3.9
	* opencv-contrib-python
	* opencv-contrib-python==4.5.5.62
	* tini
	* jq
	* curl
	* vim
	* zi
	* unzip
	* wget
	* onepanel SDK 1.0.1
打包方法：
1. to build the image, notice: please use GFW proxy to get better speed. Please do use proxy to resolve GFW issue
	* `docker build .  --build-arg http_proxy=http://172.17.0.1:1087 --build-arg https_proxy=http://172.17.0.1:1087 -t cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv:v1 --no-cache`
1. to use, run the following:
	* `docker run -it cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv:v1 bash`
1. opencv的镜像很奇怪，打好镜像后，使用时出现问题“AttributeError: module 'cv2' has no attribute 'imread'”
1. 进入镜像后，手动运行pip3 unintall opencv-python-headless 然后再运行 pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless，就正常了
1. 从正常运行的容器中导出镜像： docker commit 4a03ac55affc cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv:v1.1
1. 镜像打包完毕
