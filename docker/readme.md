# 工作正常版本：
**cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv:v1.1**
**cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv.go:v1.2**
# 镜像cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv:v1.1的规格 (1.2G):
1. conda 4.12.0
2. py3.9
4. opencv-contrib-python==4.5.5.62
5. transformers 4.22.0
6. numpy 1.23.3
7. opencv-python-headless 4.6.0.66
8. tini
9. jq
10. curl
11. vim
12. zip
13. unzip
14. wget
15. nslookup
16. onepanel SDK 1.0.1
# 镜像cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv.go:v1.2的规格(2.2G):
1. conda 4.12.0
2. py3.9
4. opencv-contrib-python==4.5.5.62
5. transformers 4.22.0
6. numpy 1.23.3
7. opencv-python-headless 4.6.0.66
8. tini
9. jq
10. curl
11. vim
12. zip
13. unzip
14. wget
15. nslookup
16. go 1.15.11 (conda activate go1.15)
17. onepanel SDK 1.0.1
# 打包方法：
1. to build the image, notice: please use GFW proxy to get better speed. Please do use proxy to resolve GFW issue
	* `docker build .  --build-arg http_proxy=http://172.17.0.1:1087 --build-arg https_proxy=http://172.17.0.1:1087 -t cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv:v1 --no-cache`
1. opencv的镜像很奇怪，打好镜像后，使用时出现问题“AttributeError: module 'cv2' has no attribute 'imread'”
1. 进入镜像后，手动运行`pip3 unintall opencv-python-headless` 然后再运行 `pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless==4.6.0.66` ，就正常了
1. 从正常运行的容器中导出镜像： docker commit 4a03ac55affc cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv:v1.1
1. 镜像打包完毕
# 使用方法
1. to use, run `docker run -it cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv:v1 bash`
