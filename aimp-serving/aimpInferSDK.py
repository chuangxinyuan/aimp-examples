from __future__ import print_function
import io
import os
import time
import base64
import json
import re
import time
import numpy as np 
import requests
import pickle
import onepanel.core.api
from onepanel.core.api.rest import ApiException
import onepanel.core.auth
from pprint import pprint

class aimpInfer:
    # MUST specify according to your situation, host请注意URL的后缀api和infer_host后面的/
    namespace = 'mp'
    model_name = 'faster-rcnn-torchserve'
    username='admin'
    token='5aed14f5bffc9f86fd0fb2745519f2ff'
    aimp_host='http://onepanel.niuhongxing.cn/api'
    infer_host='https://infer.dev.aimpcloud.cn/'

    #used by inference
    infer_host_FQDN=''
    api_access_token=''
    infer_endpoint=''

    def __init__(self):
        pass

    def getAccess(self):
        # step 1 获取用户访问onepanel的 api key 和access_token
        access_token = onepanel.core.auth.get_access_token(username=self.username, token=self.token, host=self.aimp_host)
        # Configure API key authorization: Bearer
        # Defining the host is optional and defaults to http://localhost:8888
        # See configuration.py for a list of all supported configuration parameters.
        configuration = onepanel.core.api.Configuration(
            host = self.aimp_host,
            api_key = {
            'authorization': access_token
            }
        )
        configuration.api_key_prefix['authorization'] = 'Bearer'
        # Enter a context with an instance of the API client
        # 根据APIKEY 获取 对应的token，但是目前的实现是API搭建了框架，里面并没有实现
        api_client=onepanel.core.api.ApiClient(configuration)


        # step 2 获取api access token, Create an instance of the API class
        # 因为onepanel 1.0.2实现可能不完整， AIMP0.95的suername还是admin，和对应的token
        auth_api_instance = onepanel.core.api.AuthServiceApi(api_client)
        body = onepanel.core.api.GetAccessTokenRequest(username=self.username, token=self.token) # GetAccessTokenRequest |
        try:
            auth_api_response = auth_api_instance.get_access_token(body)
            print(type(auth_api_response))
            self.api_access_token=auth_api_response.access_token
            print('---api_auth_token---')
            pprint(auth_api_response)
            print('\n')
        except ApiException as e:
            print("Exception when calling AuthServiceApi->get_access_token: %s\n" % e)

        # step 3 获取api的predict URL
        infer_api_instance = onepanel.core.api.InferenceServiceApi(api_client)
        try:
            ready = False
            while not ready:
                infer_api_response = infer_api_instance.get_inference_service(self.namespace, self.model_name)
                ready = infer_api_response.ready
                endpoint = infer_api_response.predict_url
                print('---api_response.predict_url---')
                pprint(endpoint)

                # ? is non greedy, get FQDN of predict URL
                self.infer_host_FQDN=re.findall(r"http.?//(.*?)/",endpoint)[0]
                print (self.infer_host_FQDN)
                self.infer_endpoint=re.sub(r"http.?//.*?/",self.infer_host, endpoint)
                print('\n')
        except ApiException as e:
            print("Exception when calling InferenceServiceApi->get_inference_service_status: %s\n" % e)
