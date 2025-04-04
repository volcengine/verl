import asyncio
import json
import uuid
from time import time
from typing import List, Optional, Dict, Any

from browser import KiltBrowser
from fastapi import FastAPI, HTTPException, Header, Request, Response
from pydantic import BaseModel

from verl.utils.reward_score import _default_compute_score

data = None


class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    function: Function


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Optional[str] = None
    function: FunctionCall


class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = None


class StartSampleRequest(BaseModel):
    index: int
    name: str


class StartSampleResponse(BaseModel):
    sid: str
    messages: List[Message]
    tools: List[Tool]
    metrics: Optional[Dict[str, Any]] = None


class InteractRequest(BaseModel):
    messages: List[Message]


class InteractResponse(BaseModel):
    messages: List[Message]
    finish: bool
    reward: float
    metrics: Dict[str, Any]


class CancelRequest(BaseModel):
    session_id: str
    done: bool


app = FastAPI()

# 会话管理
sessions = {}

tools = [
    Tool(
        function=Function(
            name="search",
            description="A tool for searching the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search"
                    }
                }
            }
        )),
    Tool(
        function=Function(
            name="click",
            description="A tool for clicking on a link",
            parameters={
                "type": "object",
                "properties": {
                    "link_id": {
                        "type": "integer",
                        "description": "The id of the link to click"
                    }
                }
            }
        )),
]
system_prompt = "You are a helpful assistant."

request_count = 0
start_time = time()


@app.middleware("http")
async def track_qps(request: Request, call_next):
    global request_count, start_time
    request_count += 1
    response = await call_next(request)

    elapsed_time = time() - start_time
    if elapsed_time >= 1:
        qps = request_count / elapsed_time
        print(f"QPS: {qps:.2f}")
        request_count = 0
        start_time = time()

    return response


def build_history_str(history):
    return "\n".join([f"# {msg['role']}:\n {msg['content']}\n\n" for msg in history])


@app.post("/start_sample", response_model=StartSampleResponse)
async def start_sample(request: StartSampleRequest, response: Response):
    index = request.index
    name = request.name  # not used currently

    sid = str(uuid.uuid4())
    entry = data.iloc[index]
    question = entry["question"]

    history = [Message(role="system", content=system_prompt), Message(role="user", content=question)]

    browser = KiltBrowser(
        user_prompt="",
        es_search_url=f"http://10.50.60.34:9200/kilt/_search",
        knowledge_service_url=f"http://172.18.193.64:8001/documents",
    )

    sessions[sid] = {
        "sid": sid,
        "index": index,
        "history": history,
        "browser": browser,
        "entry": entry,
    }

    metrics = {
        "search_times": 0,
        "click_times": 0,
        "observations_times": 0,
        "failed_times": 0,
    }

    response.headers["session_id"] = sid
    return StartSampleResponse(sid=sid, messages=history, tools=tools, metrics=metrics)


@app.post("/interact", response_model=InteractResponse)
async def interact(request: InteractRequest, session_id: str = Header(..., convert_underscores=False)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session_id not found")

    session = sessions[session_id]

    tool_calls = request.messages[-1].tool_calls

    if not tool_calls:
        # get reward
        answer = request.messages[-1].content
        ground_truth = session["entry"]["answer"]
        data_source = session["entry"]["data_source"]
        extra_info = session["entry"]["extra_info"]

        score = await asyncio.to_thread(
            _default_compute_score,
            data_source=data_source,
            solution_str=answer,
            ground_truth=ground_truth,
            extra_info=extra_info,
            question=system_prompt,
        )

        del sessions[session_id]

        return InteractResponse(messages=[], finish=True, reward=score, metrics={})

    # execute tool call
    browser = session["browser"]
    metrics = {"search_times": 0, "click_times": 0, "observations_times": 0, "failed_times": 0}
    for tool_call in tool_calls:
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        metrics['observations_times'] += 1

        max_retries = 3
        for i in range(max_retries):
            try:
                browser_return = await asyncio.to_thread(browser.do, name, **arguments)
                observation = browser_return["observation"]
                reason = browser_return["reason"].lower()

                if observation:
                    metrics['search_times'] += int('search' in reason)
                    metrics['click_times'] += int('click' in reason)
                    break
                else:
                    print({"info": f"Retry {i + 1} for empty observation", "observation": observation, "reason": reason})
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                import time
                print(f"API call failed: {e}")
        else:
            observation = ""
            metrics['failed_times'] += 1

    message = Message(role="tool", content=observation)

    return InteractResponse(messages=[message], finish=False, reward=0.0, metrics=metrics)


@app.post("/cancel")
async def cancel(_: Request, session_id: str = Header(..., convert_underscores=False)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session_id not found")

    del sessions[session_id]

    return {"status": "cancelled"}

@app.post("/cancel_all")
async def cancel_all(_: Request):
    global sessions
    sessions = {}
    return {"status": "cancelled_all"}


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="/workspace/dr/datasets/search_data_with_system/hotpotQA_qwen_system/train.parquet")
    args = parser.parse_args()
    data = pd.read_parquet(args.data)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# >>> data.iloc[0]
# level                                                        easy
# question        What actor best known for his portrayal of Dr....
# answer                                                Nigel Bruce
# _id                                      5a8e0bf755429917b4a5bcf6
# type                                                       bridge
# data_source                                              hotpotQA
# prompt          [{'content': '你是一个名为Qwen的人工智能助手。你的任务是针对用户的问题和要...
# ability                                                      math
# reward_model     {'ground_truth': 'Nigel Bruce', 'style': 'rule'}
# extra_info      {'answer': 'Nigel Bruce', 'index': 0, 'questio...
# Name: 0, dtype: object
