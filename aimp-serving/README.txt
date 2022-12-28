#usage:
* in this folder, run the following command:

* #hook current folder to mnt in the container which has python3.9 and opencv installed,
* #please refer to ../docker folder for details of this image
* enter each model folder to carry out the test, 模型目录，运行<模型名字-example.py>程序，查看执行效果
``` bash
:
cd /mnt/aimp-serving-test/efficientnet-v2-tfserving
python efficientnet-v2-tfserving-example.py

# real examples as the following:
root@dev-AIMP91-0:~/aimp-examples/aimp-serving# docker run -v `pwd`:/mnt -it cxyacr.azurecr.cn/shaiic/miniconda3.py39.opencv:v1.2  bash
(base) root@c2dc88cd45f3: cd /mnt/aimp-serving-test/efficientnet-v2-tfserving
(base) root@c2dc88cd45f3:/mnt/aimp-serving-test/yolox-onnx-triton# python yolox-onnx-triton-example.py

        ***************************************************************
                                  _____  __  __  _____
                           /\    |_   _||  \/  ||  __ \
                          /  \     | |  | \  / || |__) |
                         / /\ \    | |  | |\/| ||  ___/
                        / ____ \  _| |_ | |  | || |
                       /_/    \_\|_____||_|  |_||_|
        使用SDK时，请更改 aimpPredict.model_name, aimpPredict.username,
        aimpPredict.token,aimpPredict.aimp_host,aimpPredict.infer_host 为实际的值
        ***************************************************************
...............
```
