from __future__ import print_function

import io
import os
import time
import base64
import json
import time
import re

import requests
import pickle
import onepanel.core.api
from onepanel.core.api.rest import ApiException
import onepanel.core.auth
from pprint import pprint


class aimpInfer:
    # MUST specify according to your situation, host请注意URL的后缀api和infer_host后面的/
    username = 'admin'
    password = 'admin'
    host='http://portal.aimpcloud.cn/api'
    namespace = ''
    model_name = ''
    infer_host='https://infer.aimpcloud.cn/'

    #used by inference
    api_access_token=''
    infer_endpoint=''
    infer_host_FQDN=''

    def __init__(self):
        pass

   
    def getAccess(self):
        # 获取用户访问onepanel的 access_token
        # If inside of Onepanel, get mounted service account token to use as API Key
        if (os.getenv("USERNAME")):
            self.username = os.getenv("USERNAME")
        if (os.getenv("PASSWORD")):
            self.password = os.getenv("PASSWORD")
        if (os.getenv("ONEPANEL_API_URL")):
            self.host = os.getenv("ONEPANEL_API_URL")
        access_token = onepanel.core.auth.get_infer_token(username=self.username, password=self.password,host=self.host)
        self.api_access_token = access_token

        if (os.getenv("INFER_HOST")):
            self.infer_host = os.getenv("INFER_HOST")

        print('---ONEPANEL_API_URL----', self.host)
        # Configure API key authorization: Bearer
        configuration = onepanel.core.api.Configuration(
        host=self.host,
        api_key={
        'authorization': self.api_access_token
        }
        )
        configuration.api_key_prefix['authorization'] = 'Bearer'


        # Get status, endpoint
        api_client=onepanel.core.api.ApiClient(configuration)
        api_instance = onepanel.core.api.InferenceServiceApi(api_client)

        try:
           ready = False
           while not ready:
               api_response = api_instance.get_inference_service_v2(self.namespace, self.model_name)
               ready = api_response.ready
               endpoint = api_response.predict_url
               print('---api_response.predict_url---')
               pprint(endpoint)

               # ? is non greedy, get FQDN of predict URL
               self.infer_host_FQDN=re.findall(r"http.?//(.*?)/",endpoint)[0]
               print (self.infer_host_FQDN)
               self.infer_endpoint=re.sub(r"http.?//.*?/",self.infer_host, endpoint)


        except ApiException as e:
            print("Exception when calling InferenceServiceApi->get_inference_service_status: %s\n" % e)

