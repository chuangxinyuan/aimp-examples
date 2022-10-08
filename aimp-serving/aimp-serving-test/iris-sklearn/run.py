from __future__ import print_function

import io
import os
import time
import base64
import json
import time
import numpy as np 
import requests
import pickle
import onepanel.core.api
from onepanel.core.api.rest import ApiException
import onepanel.core.auth
from pprint import pprint

# MUST import AIMP python SDK
# import upper dir's python file
import sys
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferWorkFlowSDK

#start init the aimpinferWorkFlowSDK
aimpPredict=aimpInferWorkFlowSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'iris'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
infer_host_FQDN=aimpPredict.infer_host_FQDN
infer_endpoint=aimpPredict.infer_endpoint
#end init the aimpinferSDK

f = open('./iris-input.json', 'rb') #open binary file in read mode
data = f.read()

#access_token='eyJhbGciOiJSUzI1NiIsImtpZCI6IlJMYWp3WnAzcjJ5NGo3V01TNkI1ZVE3X2FBXy1wWVJ6STBERXd5SjdUbkUifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJvbmVwYW5lbCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VjcmV0Lm5hbWUiOiJhZG1pbi10b2tlbi04N3pwbCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJhZG1pbiIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6IjBkMTBlMzU3LWZkNmEtNDNkNi05ZTYyLWYwOGZhNDA5NDE3OCIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpvbmVwYW5lbDphZG1pbiJ9.jLhrCnJcDWzdaOnTR79e0UAoQMinsQ0hYMT5v2HeRRWpfaXVqo-Ewn1W7Aw51ZMZbw0DTZaxlS4LcAYWGcmhqex75xkLsyQUVcCyBvVNJaZ24FXWXfqmDWozVWFCc-92gtl4FdD8afIBBslyZJBGwZhr3aCKYYlAKQg1v8dxKIbXrdfmhIeOcTrXkrAOZqEfd70PNCfmxrEQn35zaT8eeM4KDfM6BQ0QB1HoPqKfc5heCwV3mzDKVUnABDuW6sK3QEmLbyRsNezFQCVPbz4tu9UV89g1wMY8UUJQG5AOnlbXXQpq1C67v6Sr8ZuaZ7o3R5EhbVhSdWO4w2WGN_YrPg'
headers = {
    'onepanel-access-token': access_token,
    'Content-Type': 'application/json',
    'Host': infer_host_FQDN,
}
print('---api_predict_endpoint and headers---')
print (infer_endpoint)
pprint(headers)
print('\n')
print('---Prediction RESULTS---')
# original predict URL
#r = requests.post(endpoint, headers=headers, data=data, verify=False)
# skip cert check
r = requests.post(infer_endpoint, headers=headers, data=data, verify=False)
result = r.json()
print ("预测结果说明： 0->山鸢尾（setosa）、1->变色鸢尾（versicolor）、2->维吉尼亚鸢尾（virginica）")
pprint(result)



