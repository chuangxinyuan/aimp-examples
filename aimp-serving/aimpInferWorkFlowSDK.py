from __future__ import print_function

import io
import os
import time
import base64
import json
import time

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
    infer_host='https://infer.dev.aimpcloud.cn/'

    #used by inference
    api_access_token=''
    infer_endpoint=''

    def __init__(self):
        pass

   
    def getAccess(self):
        # 获取用户访问onepanel的 access_token
        # If inside of Onepanel, get mounted service account token to use as API Key
        access_token = onepanel.core.auth.get_access_token()
        self.api_access_token = access_token

        print('---ONEPANEL_API_URL----', os.getenv('ONEPANEL_API_URL'))
        # Configure API key authorization: Bearer
        configuration = onepanel.core.api.Configuration(
        host=os.getenv('ONEPANEL_API_URL'),
        api_key={
        'authorization': access_token
        }
        )
        configuration.api_key_prefix['authorization'] = 'Bearer'


        # Get status, endpoint
        with onepanel.core.api.ApiClient(configuration) as api_client:
        api_instance = onepanel.core.api.InferenceServiceApi(api_client)

        try:
           ready = False
           while not ready:
               api_response = api_instance.get_inference_service(namespace, model_name)
               ready = api_response.ready
               endpoint = api_response.predict_url
               print('---api_response.predict_url---', endpoint)
               self.infer_endpoint=endpoint
               print('\n')
        except ApiException as e:
            print("Exception when calling InferenceServiceApi->get_inference_service_status: %s\n" % e)

