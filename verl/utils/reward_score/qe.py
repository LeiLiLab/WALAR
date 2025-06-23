import ray
import time
import requests

def request_api_wrapper(url, data, try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            return response
        except requests.RequestException as e:
            print(f"Request error, please check: {e}")
        except Exception as e:
            print(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")



@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts, labels):
    return request_api_wrapper(api_url, {"query": queries, "prompts": prompts, "labels": labels})



@ray.remote
def compute_score(prompt_str: str, solution_str: str, ground_truth: str):
    remote_rm_url = "http://localhost:5000/get_reward"  # Replace with your actual remote RM URL
    r = remote_rm_fn_ray.remote(remote_rm_url, queries=[prompt_str + solution_str], prompts=[prompt_str], labels=[ground_truth])
    r = ray.get(r)
    # print(f"r={r}")
    return r['rewards'][0]