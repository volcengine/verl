from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.sft_dataset import SFTDataset


# 事前定義されたテンプレート
CHAT_TEMPLATES = {
    "phi4-reasoning-plus": (
        "<|im_start|>system<|im_sep|>You are Phi, a language model trained by Microsoft to help users. "
        "Your role as an assistant involves thoroughly exploring questions through a systematic thinking "
        "process before providing the final precise and accurate solutions. This requires engaging in a "
        "comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, "
        "and iteration to develop well-considered thinking process. Please structure your response into two "
        "main sections: Thought and Solution using the specified format: <think> {Thought section} </think> "
        "{Solution section}. In the Thought section, detail your reasoning process in steps. Each step should "
        "include detailed considerations such as analysing questions, summarizing relevant findings, "
        "brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and "
        "revisiting previous steps. In the Solution section, based on various attempts, explorations, and "
        "reflections from the Thought section, systematically present the final solution that you deem correct. "
        "The Solution section should be logical, accurate, and concise and detail necessary steps needed to "
        "reach the conclusion. Now, try to solve the following question through the above guidelines:<|im_end|>"
        "{% for message in messages %}{% if (message['role'] == 'user') %}"
        "{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}"
        "{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>'}}"
        "{% generation %}{{message['content'] + '<|im_end|>'}}{% endgeneration %}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"
    ),
    
    "qwen3-32b": (
        "{%- if tools %}\n"
        "    {{- '<|im_start|>system\\n' }}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {{- messages[0].content + '\\n\\n' }}\n"
        "    {%- endif %}\n"
        "    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\n"
        "You are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n"
        "    {%- for tool in tools %}\n"
        "        {{- \"\\n\" }}\n"
        "        {{- tool | tojson }}\n"
        "    {%- endfor %}\n"
        "    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n"
        "{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n"
        "{%- else %}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endif %}\n"
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n"
        "{%- for message in messages[::-1] %}\n"
        "    {%- set index = (messages|length - 1) - loop.index0 %}\n"
        "    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and "
        "not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n"
        "        {%- set ns.multi_step_tool = false %}\n"
        "        {%- set ns.last_query_index = index %}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- for message in messages %}\n"
        "    {%- if message.content is string %}\n"
        "        {%- set content = message.content %}\n"
        "    {%- else %}\n"
        "        {%- set content = '' %}\n"
        "    {%- endif %}\n"
        "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
        "    {%- elif message.role == \"assistant\" %}\n"
        "        {%- set reasoning_content = '' %}\n"
        "        {%- if message.reasoning_content is string %}\n"
        "            {%- set reasoning_content = message.reasoning_content %}\n"
        "        {%- else %}\n"
        "            {%- if '</think>' in content %}\n"
        "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1]"
        ".lstrip('\\n') %}\n"
        "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
        "            {%- endif %}\n"
        "        {%- endif %}\n"
        "        {%- if loop.index0 > ns.last_query_index %}\n"
        "            {%- if loop.last or (not loop.last and reasoning_content) %}\n"
        "                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + "
        "'\\n</think>\\n\\n' + content.lstrip('\\n') }}\n"
        "            {%- else %}\n"
        "                {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        "            {%- endif %}\n"
        "        {%- else %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        "        {%- endif %}\n"
        "        {%- if message.tool_calls %}\n"
        "            {%- for tool_call in message.tool_calls %}\n"
        "                {%- if (loop.first and content) or (not loop.first) %}\n"
        "                    {{- '\\n' }}\n"
        "                {%- endif %}\n"
        "                {%- if tool_call.function %}\n"
        "                    {%- set tool_call = tool_call.function %}\n"
        "                {%- endif %}\n"
        "                {{- '<tool_call>\\n{\"name\": \"' }}\n"
        "                {{- tool_call.name }}\n"
        "                {{- '\", \"arguments\": ' }}\n"
        "                {%- if tool_call.arguments is string %}\n"
        "                    {{- tool_call.arguments }}\n"
        "                {%- else %}\n"
        "                    {{- tool_call.arguments | tojson }}\n"
        "                {%- endif %}\n"
        "                {{- '}\\n</tool_call>' }}\n"
        "            {%- endfor %}\n"
        "        {%- endif %}\n"
        "        {{- '<|im_end|>\\n' }}\n"
        "    {%- elif message.role == \"tool\" %}\n"
        "        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n"
        "            {{- '<|im_start|>user' }}\n"
        "        {%- endif %}\n"
        "        {{- '\\n<tool_response>\\n' }}\n"
        "        {{- content }}\n"
        "        {{- '\\n</tool_response>' }}\n"
        "        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n"
        "            {{- '<|im_end|>\\n' }}\n"
        "        {%- endif %}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n"
        "    {{- '<|im_start|>assistant\\n' }}\n"
        "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
        "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endif %}"
    )
}


class CustomSFTDataset(SFTDataset):
    """設定ファイルからchat_templateを読み込めるカスタムSFTデータセット"""
    
    def __init__(self, parquet_files, tokenizer, config):
        # configからchat_templateを取得
        chat_template = config.get("chat_template", None)
        
        if chat_template:
            # 事前定義されたテンプレート名の場合
            if chat_template in CHAT_TEMPLATES:
                tokenizer.chat_template = CHAT_TEMPLATES[chat_template]
            # 直接テンプレート文字列が指定された場合
            else:
                tokenizer.chat_template = chat_template
        
        # 親クラスの初期化を呼び出す
        super().__init__(parquet_files, tokenizer, config)


class CustomMultiTurnSFTDataset(MultiTurnSFTDataset):
    """設定ファイルからchat_templateを読み込めるカスタムマルチターンSFTデータセット"""
    
    def __init__(self, parquet_files, tokenizer, config):
        # configからchat_templateを取得
        chat_template = config.get("chat_template", None)
        
        if chat_template:
            # 事前定義されたテンプレート名の場合
            if chat_template in CHAT_TEMPLATES:
                tokenizer.chat_template = CHAT_TEMPLATES[chat_template]
            # 直接テンプレート文字列が指定された場合
            else:
                tokenizer.chat_template = chat_template
        
        # 親クラスの初期化を呼び出す
        super().__init__(parquet_files, tokenizer, config)