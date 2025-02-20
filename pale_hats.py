import requests
import pprint
import pandas as pd
import json
from tqdm import tqdm

query = "What is the most efficient hunter in the world?"
API_KEY = '14ff91d2ec775415624c6475d385a441'

all_results = []

for uid in tqdm(range(1024), desc="Processing UIDs"):
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    params = {"search_query": query}

    data = {
        "n_miners": 1,
        "n_results": 2,
        "max_response_time": 10,
        "target_uids": [str(uid)]
    }
    
    response = requests.post("http://0.0.0.0:42172/web_retrieval", headers=headers, params=params, json = data)
    
    try:
        result = response.json()
        result_dict = {
            'uid': uid,
            'result': response.json(),
        }
        all_results.append(result_dict)
    except (IndexError, KeyError, json.JSONDecodeError):
        all_results.append({
            'uid': uid,
            'result': None,
        })

df = pd.DataFrame(all_results, columns=['uid', 'result'])
df.to_csv("the_hats_who_are_white.csv")