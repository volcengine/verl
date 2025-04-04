      
import copy
import sys
import os
import json
import requests
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_path)
from vllm import LLM, SamplingParams

def qinyan_search(prompt, query, type="md"):
    # assert type == "md"
    # data = {
    #     "prompt": prompt,
    #     "queries": [query] if isinstance(query, str) else query
    # }
    data = {
        "q": query,
    }
    headers = {
        "Content-Type": "application/json"
    }
    url = "http://10.51.160.193:16001/search"
    response = requests.post(url, json=data, headers=headers)
    # print(response.json().keys())
    return response.json()['webPages']['value']

    
def qinyan_fetch_content(url_list, type="pa"):
    assert type in ["pa", "md"]
    headers = {
        "Content-Type": "application/json"
    }
    if type == "pa":
        data = {
            "urls": url_list
        }
        response = requests.post("http://10.51.160.193:8000/multi_browser", json=data, headers=headers)

        return {r["url"]: r["content"] if "content" in r else "" for r in response.json()}
    else:
        def fetch_single_url(url):
            try:
                data = {
                    "url": url
                }
                response = requests.post("http://10.51.160.193:8088/fetch", json=data, headers=headers)
                return url, response.json().get("md_content", "")
            except Exception as e:
                return url, ""

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(fetch_single_url, url): url for url in url_list}
            for future in concurrent.futures.as_completed(future_to_url):
                url, content = future.result()
                results[url] = content   
        return results

# seeker
class OfflineSeeker(object):
    def __init__(self, url, content_max_length=2000, content_type="pa"):
        self.url = url
        self.content_max_length = content_max_length
        self.user_query = None
        self.request_param = {
            "temperature": 0.0,
            "top_p": 0.7,
            "max_tokens": 10000,
            "stream": False,
            "stop": ["<|user|>", "<|assistant|>", "<|endoftext|>", "<|observation|>"],
            "do_sample": False,
            "total_tokens": 131072
        }
        self.content_type = content_type

    def system_prompt(self):
        system_prompt = """你是基于智谱AI公司的训练的语言模型GLM-4，作为信息筛选助手，你的任务是根据给定的搜索关键词，从提供的信息中筛选出相关且详细的信息。\n\n# 任务描述\n1. 给定信息包含多条，每条信息将提供id，种类分为以下三种之一：\n    - snippet: 仅包含摘要信息\n    - cached: 包含摘要及部分缓存正文信息\n    - full_open: 包含摘要及全部正文信息\n2. 在选择信息时，请优先考虑以下因素：\n   - 信息内容与搜索词的相关性\n   - 信息内容的实时性\n   - 信息内容的丰富度\n3. 直接返回合适信息id列表，如果信息不能回答关键词，则返回空列表。当提供的信息种类为`full_open`，请尽量选择相关的网页id列表，而不要直接返回空列表"""
        return {"role": "system", "content": system_prompt, "metadata": ""}
    
    def format_snippets(self, snippets, status):
        snippet_mds = []
        for snippet in snippets:
            if status == "snippet":
                snippet_mds.append(f"# 【{snippet['index']}†{snippet['name']}†{snippet['url']}】\n{snippet['snippet']}")
            elif status == "cached":
                snippet_mds.append(f"# 【{snippet['index']}†{snippet['name']}†{snippet['url']}】\n{snippet['snippet']}\n{snippet.get('content', '')}")
            elif status == "full_open":
                snippet_mds.append(f"# 【{snippet['index']}†{snippet['name']}†{snippet['url']}】\n{snippet['snippet']}\n{snippet['content']}")
        return "\n".join(snippet_mds)
    
    def user_prompt(self, search_query, snippet_status, snippets):
        infos = self.format_snippets(snippets, snippet_status)
        user_prompt = f"# 搜索词\n\n{search_query}\n\n# 类型\n\n{snippet_status}\n\n# 信息\n\n{infos}"
        return {"role": "user", "content": user_prompt, "metadata": ""}
    
    def call_model(self, history):

        def build_message(role, metadata, msg):
            return role + metadata + "\n" + msg
        
        prompt = ""
        for message in history:
            prompt += build_message(f"<|{message['role']}|>", message.get("metadata", ""), message["content"])

        r_p = copy.deepcopy(self.request_param)
        r_p["prompt"] = prompt + "<|assistant|>"
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.url, headers=headers, data=json.dumps(r_p))
        # print(response.text,self.url)
        return response.json()["choices"][0]["text"], response.json()["choices"][0]["stop_reason"]
    
    # @retry(10, 1)
    def process_search(self, search_query, user_query):
        # print("in process_search")
        search_res = qinyan_search(user_query, search_query, self.content_type)
        # 处理基础信息
        for i, snippet in enumerate(search_res):
            snippet["index"] = i
            snippet["url"] = snippet.pop("display_url")
            snippet["cached"] = bool(snippet.get("content", ""))
            snippet["snippet"] = snippet["snippet"][:self.content_max_length]
            if snippet.get("content", ""):
                snippet["content"] = snippet["content"][:self.content_max_length]

        # snippet
        info_levels = ["snippet", "cached", "full_open"]
        return_info = []
        for info_level in info_levels:
            messages = [self.system_prompt(), self.user_prompt(search_query, info_level, search_res)]
            # response, stop_sequence = self.call_model(messages)
            # 随机返回一个两个int的列表
            response = json.dumps([0, 1])
            ids = json.loads(response)
            if len(ids) > 0:
                return_info = [search_res[id] for id in ids]
                break

            if info_level == "cached" and len(ids) == 0:
                uncached_snippets = {snippet["url"]: snippet for snippet in search_res if not snippet.get("content", "")}
                if len(uncached_snippets) > 0:
                    content_dict = qinyan_fetch_content(list(uncached_snippets.keys()), self.content_type)
                    for url, content in content_dict.items():
                        uncached_snippets[url]["content"] = content[:self.content_max_length]

        for snippet in return_info:
            if (snippet.get("content", "") and info_level == "snippet") or (snippet.get("content", "") and not snippet["cached"] and info_level == "cached"):
                if "content" in snippet:
                    snippet.pop("content")
        # print(f"search_res: {return_info}")
        
        return return_info
    
    def search(self, search_queries, user_query):
        search_res = []
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.process_search, search_query, user_query) for search_query in search_queries]
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=100)
                    search_res.extend(result)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"search error! {e}")
        return search_res


# 串联离线模型效果
class OfflineSearch(object):
    def __init__(self, url,llm, search_engine: OfflineSeeker, content_type="pa"):
        self.url = url
        self.request_param = {
            "temperature": 0.0,
            "top_p": 0.7,
            "max_tokens": 10000,
            "stream": False,
            "stop": ["<|user|>", "<|assistant|>", "<|endoftext|>", "<|observation|>"],
            # "do_sample": False,
            # "total_tokens": 131072
        }
        self.search_engine = search_engine
        self.allowed_metatdata = ["simple_browser", "python", "生成PPT", "GenerateMindMap", "GenerateMermaidDiagram", "", "thought"]
        self.url_id_map = {}
        self.id_snippet_map = {}
        self.url_idx = 0
        self.longcite = True
        self.cogview = False
        self.content_type = content_type
        self.llm = llm

    def reset_state(self):
        self.url_id_map = {}
        self.id_snippet_map = {}
        self.url_idx = 0

    def system_prompt(self):
        system_prompt = """你是一个名为智谱清言（ChatGLM）的人工智能助手。你是基于智谱 AI 公司训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具\n\n## python\n\n当你向 `python` 发送包含 Python 代码的消息时，该代码将会在一个有状态的 Jupyter notebook 环境中执行。\n`python` 返回代码执行的输出，或在执行 60 秒后返回超时。\n`/mnt/data` 将会持久化存储你的文件。在此会话中，`python` 无法访问互联网。不要使用 `python` 进行任何网络请求或者在线 API 调用，这些在线内容的访问将不会成功。\n\n## simple_browser\n\n你可以使用 `simple_browser` 工具。该工具支持以下函数：\n`msearch(description: str, queries: list[str], recency_days: int)`：使用搜索引擎进行查询并显示结果，并在 `description` 中向用户简要描述你当前的动作。\n`open_url(url: list[str])`：打开指定的 URL。\n\n使用 `【{引用 id}†source】` 来引用内容。\n\n操作步骤：1. 使用 `msearch` 来获得信息列表; 2. 根据获得的内容进行回复。在回复中应当引用信息来源。\n 如果用户提供了 URL，也可以用 `open_url` 直接打开页面。\n如果初次搜索结果没有找到合适的信息，也可以再次使用 `msearch` 进行搜索。"""
        return {"role": "system", "content": system_prompt, "metadata": ""}

    # def call_model(self, history):

    #     def build_message(role, metadata, msg):
    #         return role + metadata + "\n" + msg
        
    #     prompt = ""
    #     for message in history:
    #         if message is not None:
    #             prompt += build_message(f"<|{message['role']}|>", message.get("metadata", ""), message["content"])

    #     r_p = copy.deepcopy(self.request_param)
    #     r_p["prompt"] = prompt + "<|assistant|>"
    #     r_p["model"] = "/workspace/lurui-yun/deep_research/glm-train-dev/checkpoint/9b-sft/9b_simple_browser_0218/hf_0000308"
    #     headers = {"Content-Type": "application/json"}
    #     response = requests.post(self.url, headers=headers, data=json.dumps(r_p))
    #     # print(response.json()["choices"][0]["text"])   
    #     return response.json()["choices"][0]["text"], response.json()["choices"][0]["stop_reason"]
    def call_model(self, history):
        def build_message(role, metadata, msg):
            return role + metadata + "\n" + msg
        
        prompt = ""
        for message in history:
            if message is not None:
                prompt += build_message(f"<|{message['role']}|>", message.get("metadata", ""), message["content"])

        r_p = copy.deepcopy(self.request_param)
        r_p["prompt"] = prompt + "<|assistant|>"

        # 创建SamplingParams对象
        sampling_params = SamplingParams(
            temperature=r_p.get("temperature", 0.0),
            top_p=r_p.get("top_p", 0.7),
            max_tokens=r_p.get("max_tokens", 10000),
            stop=r_p.get("stop", ["<|user|>", "<|assistant|>", "<|endoftext|>", "<|observation|>"]),
        )

        # 调用本地LLM模型进行推理
        outputs = self.llm.generate(prompts=r_p["prompt"], sampling_params=sampling_params)
        return outputs[0].outputs[0].text, outputs[0].outputs[0].stop_reason

        # return response["choices"][0]["text"], response["choices"][0]["stop_reason"]


    def parse_response(self, response:str):
        try:
            metadata = response.split("\n")[0]
            content = response[len(metadata+"\n"):]

            if metadata in ["simple_browsersimple_browser"]:
                metadata="simple_browser"
                
            assert metadata in self.allowed_metatdata, f"metadata:{metadata},content:{content},response:{response} is not allowed"
        except Exception as e:
            print(f"parse_response error! {e}")
            return response
        return {"role": "assistant", "content": content, "metadata": metadata,"loss": True}
    
    def format_snippet(self, snippet):
        title = f"# 【{snippet['index']}†{snippet['name']}†{snippet['url']}】"
        snippet_md = f"{title}\n{snippet['snippet']}"
        if "content" in snippet and snippet["content"]:
            snippet_md += f"\n{snippet['content']}"
        return snippet_md

    def function_call(self, method, params):
        observations = []
        if method == "open_url":
            url_list = params
            url_contents = qinyan_fetch_content(url_list, self.content_type)

            url_mds = []
            for url, content in url_contents.items():
                url_mds.append(f"URL: {url}\nURL_CONTENT: {content}")
            observations.append({
                "metadata": "browser_result",
                "content": "\n".join(url_mds),
                "loss": False,
            })
        elif method == "msearch":
            search_queries = params["queries"]
            msearch_res = self.search_engine.search(search_queries, self.user_query)
            # print(f"msearch_res: {msearch_res}")
            

            snippet_mds = []
            for snippet in msearch_res:
                if snippet["url"] not in self.url_id_map:
                    self.url_id_map[snippet["url"]] = self.url_idx
                    snippet["index"] = self.url_idx
                    self.id_snippet_map[snippet["index"]] = copy.deepcopy(snippet)
                    self.url_idx += 1
                    snippet_info=self.format_snippet(snippet)
                    observations.append({
                        "metadata": f"quote_result[{snippet['index']}†source]",
                        "content": snippet_info,
                        "loss": False,
                    })
        return [{
                "role": "observation",
                "metadata": ob["metadata"],
                "content": ob["content"],
                "loss": False,
            } for ob in observations]

    def parse_params(self, metadata, content):
        if metadata == "simple_browser":
            function_heads = ["msearch", "open_url"]
            for head in function_heads:
                if content.startswith(head):
                    post_json_str = content[(len(head) + 1):-1].replace("description=","\"description\":").replace("queries=","\"queries\":").replace("ids=", "\"ids\":").replace("recency_days=", "\"recency_days\":")
                    if head in ["msearch"]:
                        post_json_str = "{" + post_json_str + "}"
                    return head, json.loads(post_json_str)
        elif metadata in ["python"]:
            return metadata, content
        elif metadata in ["生成PPT", "GenerateMindMap", "GenerateMermaidDiagram"]:
            return metadata, json.loads(content)
        elif metadata in ["thought"]:
            return metadata, content
        else:
            raise NotImplementedError(f"unkonw method! metadata:{metadata},content:{content}")

    def chat(self, his, max_steps):
        self.reset_state()
        sys_prompt = self.system_prompt()
        sys_prompt["loss"] = False
        history = [sys_prompt]
        history.extend(his)
        self.user_query = his[-1]["content"]
        online_infer_index = len(history)
        is_response = False
        step = 0
        while not is_response:
            step += 1
            if max_steps is not None and step > max_steps:
                return history, True
            model_content, stop_token = self.call_model(history)
            model_output = self.parse_response(model_content)
            history.append(model_output)
            # print(f"model_output: {model_output}")
            # if model_output["metadata"] == "" and "user" in stop_token:
            if model_output["metadata"] == "" and stop_token is None:
                is_response = True
            else:
                if model_output["metadata"] != "":
                    method, params = self.parse_params(model_output["metadata"], model_output["content"])
                    ob = self.function_call(method, params)
                    # print(f"ob: {ob}")
                    for o in ob:
                        if o is not None:
                            history.append(o)
        print("history:", history)
        return history, False

def try_case(query, search_engine_url, main_chat_url, max_steps=10, content_type="pa",model_path="/data/o1-cloud/lurui/checkpoint/9b_simple_hf_epoch_1_0218"):
    llm = LLM(model=model_path,trust_remote_code = True)
    search_engine = OfflineSeeker(search_engine_url, content_type=content_type)
    main_chat = OfflineSearch(main_chat_url, llm, search_engine=search_engine, content_type=content_type)
    his,overflow = main_chat.chat([{
        "role": "user",
        "content": query
    }], max_steps)
    return his

if __name__ == "__main__":
    content_type = "pa"
    max_steps = 20
    search_engine_url = "http://172.18.75.5:9090/v1/completions"
    main_chat_url = "http://172.18.80.43:8000/v1/completions"
    # try_case("""基于钠的光影端口""", search_engine_url, main_chat_url, max_steps=max_steps, content_type=content_type)
    # try_case("""月之暗面的CEO本科导师创办的公司是什么？""", search_engine_url, main_chat_url, max_steps=max_steps, content_type=content_type,model_path="/data/o1-cloud/lurui/checkpoint/9b_simple_hf_epoch_1_0218")
    his = try_case("""Which magazine was started first Arthur's Magazine or First for Women?""", search_engine_url, main_chat_url, max_steps=max_steps, content_type=content_type,model_path="/data/o1-cloud/lurui/checkpoint/9b_simple_hf_epoch_1_0218")
    print(his)
    