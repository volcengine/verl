# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapted from https://python.langchain.com/docs/tutorials/sql_qa/
as well as https://langchain-ai.github.io/langgraph/tutorials/sql-agent/
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from typing import Any, Literal, Optional

import agentlightning
import dotenv
import termcolor
from langchain.chat_models import init_chat_model
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from spider_eval.exec_eval import eval_exec_match

agentlightning.configure_logger()

logger = agentlightning.configure_logger(name=__name__)


WRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
     Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
GENERATED QUERY
```
""".strip(),
        ),
        ("user", "Question: {input}"),
    ]
)


CHECK_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Explicit query execution failures
- Clearly unreasoable query execution results

## Table Schema ##

{table_info}

## Output Format ##

If any mistakes from the list above are found, list each error clearly.
After listing mistakes (if any), conclude with **ONE** of the following exact phrases in all caps and without surrounding quotes:
- If mistakes are found: `THE QUERY IS INCORRECT.`
- If no mistakes are found: `THE QUERY IS CORRECT.`

DO NOT write the corrected query in the response. You only need to report the mistakes.
""".strip(),  # noqa: E501
        ),
        (
            "user",
            """Question: {input}

Query:

```{dialect}
{query}
```

Execution result:

```
{execution}
```""",
        ),
    ]
)


REWRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
Rewrite the previous {dialect} query to fix errors based on the provided feedback.
The goal is to answer the original question.
Make sure to address all points in the feedback.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
REWRITTEN QUERY
```
""".strip(),
        ),
        (
            "user",
            """Question: {input}

## Previous query ##

```{dialect}
{query}
```

## Previous execution result ##

```
{execution}
```

## Feedback ##

{feedback}

Please rewrite the query to address the feedback.""",
        ),
    ]
)


class State(MessagesState):
    question: str
    query: str
    execution: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[BaseMessage]


class SQLAgent:
    def __init__(
        self,
        db: str,
        max_turns: int = 5,
        debug: bool = False,
        db_schema: str | None = None,
        endpoint: str | None = None,
        verl_replacement: dict | None = None,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ):
        self.db = SQLDatabase.from_uri(db)
        self.db_schema = db_schema
        self.debug = debug
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
        if verl_replacement is not None:
            self.model_name = verl_replacement["model"]
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=verl_replacement["temperature"],
                max_retries=0,
                max_tokens=2048,
            )
        else:
            self.model_name = os.environ.get("MODEL", "gpt-4.1-mini")
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"],
                temperature=0,
                max_retries=1,
                max_tokens=2048,
            )

    def get_table_info(self) -> str:
        """Get the table information in a human-readable format."""
        try:
            table_info = self.db.get_table_info()
            if len(table_info) > self.table_info_truncate:
                table_info = table_info[: self.table_info_truncate] + "\n... (truncated)"
            return table_info
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            if self.db_schema:
                if len(self.db_schema) > self.table_info_truncate:
                    return self.db_schema[: self.table_info_truncate] + "\n... (truncated)"
                return self.db_schema
            return "No schema available."

    def invoke_prompt(self, prompt: Any) -> BaseMessage:
        if self.debug:
            for message in prompt.messages:
                termcolor.cprint(message.pretty_repr(), "blue")

        try:
            result = self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Failed to invoke prompt: {e}")
            # FIXME: fallback to create a random trajectory
            result = self.llm.invoke([HumanMessage(content="Please create a random SQL query as an example.")])

        if self.debug:
            termcolor.cprint(result.pretty_repr(), "green")

        return result

    def truncate_execuion(self, execution: str) -> str:
        """Truncate the execution result to a reasonable length."""
        if len(execution) > self.execution_truncate:
            return execution[: self.execution_truncate] + "\n... (truncated)"
        return execution

    def parse_query(self, message: BaseMessage) -> str | None:
        result = None
        for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):
            result = match.group(1).strip()
        return result

    def write_query(self, state: State):
        """Generate SQL query to fetch information."""
        prompt = WRITE_QUERY_PROMPT.invoke(
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)

        query = self.parse_query(result) or result.content

        return {
            **state,
            "query": query,
            "num_turns": 1,
            "messages": [*prompt.messages, result],
        }

    def execute_query(self, state: State) -> State:
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        execution_result = execute_query_tool.invoke(state["query"])
        if not isinstance(execution_result, str):
            # Convert to string if it's not already
            execution_result = str(execution_result)
        if self.debug:
            termcolor.cprint(execution_result, "yellow")
        return {**state, "execution": execution_result}

    def check_query(self, state: State) -> State:
        """Check the SQL query for correctness."""
        prompt = CHECK_QUERY_PROMPT.invoke(
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execuion(state["execution"]),
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)

        res = {
            **state,
            "feedback": result.content,
            "messages": [*state.get("messages", []), *prompt.messages, result],
        }
        return res

    def rewrite_query(self, state: State) -> State:
        """Rewrite SQL query if necessary."""
        prompt = REWRITE_QUERY_PROMPT.invoke(
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execuion(state["execution"]),
                "feedback": state["feedback"],
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)

        rewritten_query = self.parse_query(result)

        return {
            **state,
            "query": rewritten_query or state["query"],
            "num_turns": state.get("num_turns", 0) + 1,
            "messages": [*prompt.messages, result],  # clear previous prompts
        }

    def should_continue(self, state: State) -> Literal[END, "rewrite_query"]:  # type: ignore
        """Determine if the agent should continue based on the result."""
        if state["messages"] and isinstance(state["messages"][-1], BaseMessage):
            last_message = state["messages"][-1]
            if "THE QUERY IS CORRECT" in last_message.content:
                if "THE QUERY IS INCORRECT" in last_message.content:
                    # Both correct and incorrect messages found
                    # See which is the last one
                    correct_index = last_message.content.rfind("THE QUERY IS CORRECT")
                    incorrect_index = last_message.content.rfind("THE QUERY IS INCORRECT")
                    if correct_index > incorrect_index:
                        return END
                else:
                    return END

        if state.get("num_turns", 0) >= self.max_turns:
            return END

        return "rewrite_query"

    def graph(self) -> CompiledStateGraph[State]:
        builder = StateGraph(State)
        builder.add_node(self.write_query)
        builder.add_node(self.execute_query)
        builder.add_node(self.check_query)
        builder.add_node(self.rewrite_query)

        builder.add_edge(START, "write_query")
        builder.add_edge("write_query", "execute_query")
        builder.add_edge("execute_query", "check_query")
        builder.add_conditional_edges(
            "check_query",
            self.should_continue,
        )
        builder.add_edge("rewrite_query", "check_query")

        return builder.compile()


def evaluate_query(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    # TODO(yuge): Maybe we can evaluate intermediate queries and assign more precise rewards.

    # included in the original evaluation script
    # query = query.replace("value", "1")

    try:
        database = os.path.abspath(database)
        if not os.path.exists(database):
            raise FileNotFoundError(f"Database file {database} does not exist.")

        # Parameters following the default setting
        exec_score = eval_exec_match(
            db=database,
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
        if exec_score == 1:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        if raise_on_error:
            raise
        else:
            logger.exception(f"Error evaluating query: {e}")
            return 0.0


class LitSQLAgent(agentlightning.LitAgent):
    def __init__(
        self,
        trained_agents: Optional[str] = r"write",
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.spider_dir = os.environ.get("VERL_SPIDER_DATA_DIR", "data")
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate

    def _execute_rollout(
        self, sample: dict[str, Any], *, resources: agentlightning.NamedResources, rollout_id: str, is_training: bool
    ) -> float | None:
        question = sample["question"]
        start_time = time.time()
        llm: agentlightning.LLM = resources["main_llm"]

        if is_training:
            original_db_path = os.path.join(self.spider_dir, "database", sample["db_id"], sample["db_id"] + ".sqlite")
        else:
            original_db_path = os.path.join(
                self.spider_dir, "test_database", sample["db_id"], sample["db_id"] + ".sqlite"
            )
        ground_truth = sample["query"]

        if not os.path.exists(original_db_path):
            logger.error(f"Database {original_db_path} does not exist. Skipping.")
            return None

        schema_path = os.path.join(os.path.dirname(original_db_path), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path) as f:
                schema = f.read()
        else:
            logger.error("Schema file not found: %s", schema_path)
            schema = "No schema available."

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)
            logger.info(f"[Rollout {rollout_id}] Question: {question}")
            logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

            # Run the agent
            agent = SQLAgent(
                "sqlite:///" + db_path,
                max_turns=self.max_turns,
                table_info_truncate=self.table_info_truncate,
                execution_truncate=self.execution_truncate,
                debug=False,
                db_schema=schema,
                endpoint=llm.endpoint,
                verl_replacement=(
                    {"model": llm.model, **llm.sampling_parameters}
                    if is_training
                    else {
                        "model": llm.model,
                        "temperature": (
                            self.val_temperature
                            if self.val_temperature is not None
                            else llm.sampling_parameters.get("temperature", 0.0)
                        ),
                    }
                ),
            ).graph()
            try:
                result = agent.invoke(
                    {"question": question},
                    {"callbacks": [self.tracer.get_langchain_callback_handler()], "recursion_limit": 100},
                )
            except Exception as e:
                logger.exception(f"[Rollout {rollout_id}] Error during agent invocation: {e}")
                return

            logger.info(f"[Rollout {rollout_id}] Generated Query: {result['query']}")

        end_time_rollout = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)

            reward = evaluate_query(result["query"], ground_truth, db_path, raise_on_error=False)
            logger.info("[Rollout %s] Reward: %s", rollout_id, reward)

        end_time_eval = time.time()

        logger.info("[Rollout %s] Time taken for rollout: %.2f seconds", rollout_id, end_time_rollout - start_time)
        logger.info(
            "[Rollout %s] Time taken for evaluation: %.2f seconds", rollout_id, end_time_eval - end_time_rollout
        )

        return reward

    def training_rollout(self, task: Any, rollout_id: str, resources: agentlightning.NamedResources) -> Any:
        return self._execute_rollout(task, resources=resources, rollout_id=rollout_id, is_training=True)

    def validation_rollout(self, task: Any, rollout_id: str, resources: agentlightning.NamedResources) -> Any:
        return self._execute_rollout(task, resources=resources, rollout_id=rollout_id, is_training=False)


def spider_dev_data():
    # Read from dev.parquet
    import pandas as pd

    spider_dev_data_path = os.path.join(os.environ.get("VERL_SPIDER_DEV_DATA", "data"), "dev.parquet")
    if not os.path.exists(spider_dev_data_path):
        raise FileNotFoundError(f"Spider dev data file {spider_dev_data_path} does not exist.")
    df = pd.read_parquet(spider_dev_data_path)
    if "OPENAI_API_BASE" not in os.environ:
        logger.warning(
            "Environment variable OPENAI_API_BASE is not set. Using default value 'https://api.openai.com/v1'."
        )
        openai_api_base = "https://api.openai.com/v1"
    else:
        openai_api_base = os.environ["OPENAI_API_BASE"]

    resource = {
        "main_llm": agentlightning.LLM(
            model="gpt-4.1-nano",
            endpoint=openai_api_base,
            sampling_parameters={
                "temperature": 0.0,
            },
        )
    }
    return agentlightning.DevTaskLoader(df.head(10).to_dict(orient="records"), resource)


if __name__ == "__main__":
    dotenv.load_dotenv()
    agent, trainer = agentlightning.lightning_cli(LitSQLAgent, agentlightning.Trainer)
    trainer.fit(agent, os.environ["VERL_API_BASE"], spider_dev_data())
