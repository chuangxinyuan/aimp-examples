# 上海仪电人工智能中台说明
* 上海仪电人工智能中台v1.0的AI模型推理组件是基于[kfserving 0.6.0版本](https://github.com/chuangxinyuan/aimp-kfserving/tree/release-0.6)。
## AIMP公开模型列表

|名称|描述| 服务地址  |预测器| 模型格式| 模型文件下载地址| 版本| 文档 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|efficientnet-v2-simple-tfserving|-|-| [Triton Inference Server](https://github.com/triton-inference-server/server) | [TensorFlow,TorchScript,ONNX,TensorRT](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_repository.html)| v2 |[Compatibility Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)| [README](./aimp-serving/aimp-serving-test/efficientnet-v2-simple-tfserving/READEME.md) |
|efficientnet-v2-tfserving|-|-| [TFServing](https://www.tensorflow.org/tfx/guide/serving) | [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model) | v1 ||[README](./aimp-serving/aimp-serving-test/efficientnet-v2-tfserving/READEME.md)   |
|faster-rcnn-torchserve|-|-| [TorchServe](https://pytorch.org/serve/server.html) | [Eager Model/TorchScript](https://pytorch.org/docs/master/generated/torch.save.html) | v1 | 0.3.0 |[README](./aimp-serving/aimp-serving-test/faster-rcnn-torchserve/READEME.md)  |
|iris-sklearn|-|-| [SKLearn KFServer](https://github.com/kubeflow/kfserving/tree/master/python/sklearnserver) | [Pickled Model](https://scikit-learn.org/stable/modules/model_persistence.html) 文件后缀为joblib| v1 | 0.20.3 | [README](./aimp-serving/aimp-serving-test/iris-sklearn/READEME.md) |
|yelp-polarity-triton|-|-| [SKLearn KFServer](https://github.com/kubeflow/kfserving/tree/master/python/sklearnserver) | [Pickled Model](https://scikit-learn.org/stable/modules/model_persistence.html) 文件后缀为joblib| v1 | 0.20.3 | [README](./aimp-serving/aimp-serving-test/yelp-polarity-triton/READEME.md) |
|yolov5s-tfserving|-|-| [SKLearn KFServer](https://github.com/kubeflow/kfserving/tree/master/python/sklearnserver) | [Pickled Model](https://scikit-learn.org/stable/modules/model_persistence.html) 文件后缀为joblib| v1 | 0.20.3 | [README](./aimp-serving/aimp-serving-test/yolov5s-tfserving/READEME.md) |
|yolox-onnx-triton|-|-| [SKLearn KFServer](https://github.com/kubeflow/kfserving/tree/master/python/sklearnserver) | [Pickled Model](https://scikit-learn.org/stable/modules/model_persistence.html) 文件后缀为joblib| v1 | 0.20.3 | [README](./aimp-serving/aimp-serving-test/yolov5s-tfserving/READEME.md)   |

原生的功能和支持的AI模型类型列表和相应的例子请参见：
[model servers](./docs/samples/README.md)

# 模型推理服务使用方式
* 中台推理服务的地址是 https://infer.dev.aimpcloud.cn (实际的地址根据部署的情况不同会有差异，实际的地址请向管理员索取）。

* 仪电人工智能中台的地址和推理服务地址是不同的地址。如下示例中，假定中台的地址为http://onepanel.niuhongxing.cn ，推理服务的地址是 https://infer.dev.aimpcloud.cn。 则推理服务的使用方法如下：

## 准备工作：

* 仪电人工智能中台的用户(username)和密码（token）
* AIMP和推理服务工作正常
* 推理服务所在的名字空间(namespace)和模型的名字(name)。
``` bash
# 获取AIMP某个名字空间(例如名字空间为mp）下推理服务URL的方法
root@ai-mp-1:~# for i in `k get inferenceservices.serving.kubeflow.org -n mp --no-headers|awk '{print$1}'`;do k get inferenceservices.serving.kubeflow.org $i -n mp -o jsonpath='{.status.address.url}';echo "" ;done
http://efficientnet-v2-small-tfserving.mp.svc.cluster.local/v1/models/efficientnet-v2-small-tfserving:predict
http://efficientnet-v2-tfserving.mp.svc.cluster.local/v1/models/efficientnet-v2-tfserving:predict
http://faster-rcnn-torchserve.mp.svc.cluster.local/v1/models/faster-rcnn-torchserve:predict
http://faster-rcnn-torchserve-niu.mp.svc.cluster.local/v1/models/faster-rcnn-torchserve-niu:predict
http://iris.mp.svc.cluster.local/v1/models/iris:predict
http://iris2.mp.svc.cluster.local/v1/models/iris2:predict
http://iris3.mp.svc.cluster.local/v1/models/iris3:predict
http://yelp-polarity-triton.mp.svc.cluster.local/v2/models/yelp-polarity-triton/infer
```
## 1. REST API方式 (推荐)
* REST API方式比较灵活，直接，可以使用命令行，JAVA，GO等语言实现，并且易于调试
### REST API调用序列如下
* [API详细参考和示例： /apis/v1beta/service/{name} - AIMPInferExample (apifox.cn)](https://www.apifox.cn/apidoc/project-1485755/api-35503298)， inferExample目录中的3个API。或者参考文档 [AIMP_INFER_REST_APIs.html](./aimp-serving/AIMP_INFER_REST_APIs.html)
* 端点中的地址请替换为中台的实际地址，注意后面有/api

1. 获取推理服务的access_token： REST服务端点：http://onepanel.niuhongxing.cn/api ， POST /apis/v1beta1/auth/get_access_token ，body主要两个参数 ，aimp的用户名和token（从你的租户管理员获取相应的用户名和对应的token）， 返回 access_token。
2. 获取推理服务的原始URL：REST服务端点：http://onepanel.niuhongxing.cn/api 。GET /apis/v1beta1/{namespace}/inferenceservice/{name}，header主要两个参数，namespace和 模型的名字name,返回原始的推理服务URL
3. 进行相应的推理服务调用：REST服务端点：https://infer.dev.aimpcloud.cn ， POST /v1/models/iris:predict，header 中主要2个参数，onepanel_access_token（上述API调用返回的 access_token)和 HOST（原始的推理服务URL中截取的主机地址）
   * 以curl命令方式示例： curl -k -v -H 'Host: iris--mp.niuhongxing.cn' https://infer.dev.aimpcloud.cn:443/v1/models/iris:predict -d @./iris-input.json, 其中-k 是忽略证书错误

## 2. python SDK 方式
* [参考：SDK文档](https://github.com/onepanelio/python-sdk)
* 必须是python 3 版本，python 2.6不支持，安装SDK `pip install onepanel-sdk`
### 在AIMP工作流内部使用SDK
* 工作流内部使用时，可以从上下文中获取用户名和对应的token，并且也不用显式的指定ONEPANEL_API_URL，可以从环境变量中获得。
* SDK 代码[aimpInferWorkFlowSDK.py](./aimp-serving/aimpInferWorkFlowSDK.py)
* SDK 使用方法参考 [run.py](./aimp-serving/aimp-serving-test/efficientnet-v2-tfserving/run.py)
``` python
    # 输入参数3个
    # MUST specify according to your situation, host请注意URL的后缀api和infer_host后面的/
    namespace = '根据实际情况填写'
    model_name = '根据实际情况填写'
    infer_host='https://根据实际情况填写/'

    # 输出参数2个
    # used by inference
    api_access_token=''
    infer_endpoint=''
```
### 在AIMP外部使用SDK
* 工作流内部使用时，需要指定AIMP的用户名和对应的token，还有AIMP的URL
* SDK 代码[aimpInferSDK.py](./aimp-serving/aimpInferSDK.py)
* SDK 使用方法参考 [efficientnet-v2-tfserving-test.py](./aimp-serving/aimp-serving-test/efficientnet-v2-tfserving/efficientnet-v2-tfserving-example.py)
``` python
    # SDK输入参数5个
    # MUST specify according to your situation, host请注意URL的后缀api和infer_host后面的/
    namespace = '根据实际情况填写'
    model_name = '根据实际情况填写'
    username='根据实际情况填写'
    token='根据实际情况填写'
    aimp_host='http://根据实际情况填写/api'
    infer_host='https://根据实际情况填写/'

    # 输出参数3个
    #used by inference
    infer_host_FQDN=''
    api_access_token=''
    infer_endpoint=''
```

# 模型部署和调试
## 模型部署yaml文件编写
* 登陆中台后，编写模型部署的yaml文件，然后在中台的模型发布页面中进行发布，模型部署yaml文件注意如下关键字段：
1. name: 模型的名字很关键，是做为模型推理服务URL的一部分
2. namespace: 模型部署到哪个名字空间中
3. 模型的predictor的类型，是tensorflow，sklearn等
4. runtimeVersion: predictor的版本 ，最好和kfserving-system名字空间下的inferenceservice-config configmap 中定义的版本保持一致。
5. storageUri: 是模型文件的下载链接，AIMP的模型都是从azure的blob存储的文件链接
6. request: 使用的资源数量，
7. 其他字段比如nodeaffinity等可以查看InferenceService 支持的字段
8. 例子可以参考aimp-serving目录下的模型中的部署定义文件

* 模型部署
1. 中台中点击"模型->创建模型服务"
![图片](./pics/modelUI.jpg)
2. 把模型部署yaml文件粘贴到窗口，然后点击创建
![图片](./pics/modelUI2.jpg)

## 模型部署调试方法
* 模型部署后，使用命令 `kubectl get pod -n <namepace>`, 观察部署的模型对应的pod的状态
* 使用命令 `kubectl logs <pod name> -n <namespace> --all-containers`，查看pod中所有容器的状态
* 使用命令 `kubectl exec -it <pod name> -n <namespace> -c <container name> -- bash` 进入容器观察运行的情况
* 使用命令 `kubectl get isvc -n <namespace>`查看模型服务的状态和url等
## 模型的结构和打包方法
# 模型仓库
## 模型仓库目录结构
* AIMP模型仓库位于azure cn 的对象存储中，目录结构如下
```
<models>
   <cv-models>/
      <efficient-v2-small-tfserving.zip>
      <efficientnet-v2-tfserving.zip>
      <faster-rcnn-pytorch.zip>
      ...
   <nlp-models>/
      <yelp-polarity-triton.zip>
      ...
   ...
```

# 注意
* kfserving 使用了knative来管理微服务， kanative在拉取镜像的时候，会把镜像的tag转换成镜像的sha256数字（[详细参见tag resolution](https://knative.dev/docs/serving/tag-resolution/) ），官方的镜像针对镜像一般会存储tag和对应的sha256，这样在使用的时候不会出问题。但是自定义的镜像一般只会推送 tag，不会把sha256也推送到仓库，因此，会出现镜像拉取不下来的情况，这种情况需要配置如下参数：


```bash
 #tag和sha256并存的例子：
 nvcr.io/nvidia/tritonserver@sha256:28a458eac4d888329c9a7420032f52be27fd75fef670c99c598bca76433341c0
 nvcr.io/nvidia/tritonserver:latest
 
 # Skip for `nvcr.io` which requires auth to resolve triton inference server image digest
kubectl patch cm config-deployment --patch '{"data":{"registriesSkippingTagResolving":"nvcr.io"}}' -n knative-serving
```
---
# 参考
* 模型训练，打包，上线，原理详细说明，请参考[ kfserving 0.6版本官方文档](https://github.com/chuangxinyuan/aimp-kfserving/tree/release-0.6)
* 特别是[参考案例](https://github.com/chuangxinyuan/aimp-kfserving/tree/release-0.6/docs/samples/v1beta1) 
