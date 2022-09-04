import os
import time
import base64
import json
import time
import requests
import onepanel.core.api
from onepanel.core.api.rest import ApiException
import onepanel.core.auth
from transformers import BertTokenizer
import requests



# If inside of Onepanel, get mounted service account token to use as API Key
access_token = onepanel.core.auth.get_access_token()

print('---ONEPANEL_API_URL----', os.getenv('ONEPANEL_API_URL'))
# Configure API key authorization: Bearer
configuration = onepanel.core.api.Configuration(
    host=os.getenv('ONEPANEL_API_URL'),
    api_key={
        'authorization': access_token
    }
)
configuration.api_key_prefix['authorization'] = 'Bearer'


namespace = 'mp'
model_name = 'yelp-polarity-triton'



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
            time.sleep(1)
    except ApiException as e:
        print("Exception when calling InferenceServiceApi->get_inference_service_status: %s\n" % e)




tokenizer = BertTokenizer.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")
inputs = tokenizer(["Hello, my dog is cute"])


data = {
   "inputs":[
   {
    "name": "input_0",
    "shape": [1,8],
    "datatype": "INT32",
    "data": inputs['input_ids']
   }
   ]
}

headers = {
    'onepanel-access-token': access_token
}

r = requests.post(endpoint,headers=headers, json=data)

result = r.json()

print('prediction probs:  ', result['outputs'][0]['data'])