# DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning

This directory contains the implementation for reproducing the DeepEyes paper within the verl framework, supporting multi-turn visual tool calls. This implementation is based on the original [DeepEyes paper](https://arxiv.org/abs/2505.14362) and its [official implementation](https://github.com/Visual-Agent/DeepEyes), integrated with the multi-modal and multi-turn capabilities of the verl framework.

## Reproducing the Experiment

> **Note on the 'Chart' Dataset:**
> 
> The provided preprocessing script intentionally excludes `data_v0.8_visual_toolbox_v2.parquet`, which contains the 'Chart' data. This subset consists of very high-resolution images, often resembling large figures composed of multiple sub-plots, much like those found in academic papers.
>
> Consequently, even after using the zoom-in tool, the resulting cropped images remain large. This poses a significant risk of causing Out-of-Memory (OOM) errors, which can abruptly terminate the training process. 
> 
> **We strongly recommend against training on the 'Chart' dataset on a single node.**

> **Note on the 'thinklite' Dataset:**
> Many images in the `thinklite` dataset have a very low resolution, with either a height or width below 28 pixels. This fails to meet the minimum input size required by the Qwen-2.5VL image processor and would cause errors during data loading.
>
> To mitigate this, we upscale these low-resolution images to satisfy the processor's requirements. However, please be aware that because the original resolution is low, subsequent `crop` operations by the zoom-in tool might frequently trigger exceptions, which could in turn affect the model's tool-use performance.

First, launch an inference service to act as a judge for reward calculation. You can use the following script as a reference:

```bash
python -m sglang.launch_server --model-path /path/to/Qwen2.5-72B-Instruct \
    --port 18901 \
    --tp-size 8 \
    --context-length 32768 \
    --trust-remote-code \
    --log-requests false
```

Next, you can start the training:

```bash
bash recipe/deepeyes/run_deepeyes_grpo.sh
```

## Performance

![score](https://private-user-images.githubusercontent.com/82520804/474784419-b13f4f72-bb3a-4281-a43b-1f34a9037c0c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ0NTQxMTMsIm5iZiI6MTc1NDQ1MzgxMywicGF0aCI6Ii84MjUyMDgwNC80NzQ3ODQ0MTktYjEzZjRmNzItYmIzYS00MjgxLWE0M2ItMWYzNGE5MDM3YzBjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODA2VDA0MTY1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJjNGMxMjhiOGM4MTNhYTEzYTE2MTYzY2ZjYWRhNmEzMmVjNjUxOGI3MTgzOGQyM2ZmOWJlYTZlNDYzYzU0ZDkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.qTDX-3fyLHWdeFh9o4b6nIAB57bT0XyLjKXhNV6k5nA)

![entropy](https://private-user-images.githubusercontent.com/82520804/474785253-752106a9-e25d-4b44-aef9-1ac98015d05c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ0NTQxMTMsIm5iZiI6MTc1NDQ1MzgxMywicGF0aCI6Ii84MjUyMDgwNC80NzQ3ODUyNTMtNzUyMTA2YTktZTI1ZC00YjQ0LWFlZjktMWFjOTgwMTVkMDVjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODA2VDA0MTY1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM4OGQ2ZGI3M2JlYWE4YTQyMzIxMWYxMzZhNDBmNmYxNzcwNDgxNThiZDRiMzQyYzUwZjc3OWE4YzdhYWEwMWUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.PhimMTxXXEtMLPGzejPQuw-Ul0As8ey-hyy1qkeABIQ)

![num_turns](https://private-user-images.githubusercontent.com/82520804/474785462-c99c7952-14db-485a-acd2-14e5956ecc34.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ0NTQxMTMsIm5iZiI6MTc1NDQ1MzgxMywicGF0aCI6Ii84MjUyMDgwNC80NzQ3ODU0NjItYzk5Yzc5NTItMTRkYi00ODVhLWFjZDItMTRlNTk1NmVjYzM0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODA2VDA0MTY1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJkNWYwMGVjOWM4NDVhZTkzZWI5NWMzMGVjZTcyZGM2NDExY2FmYTBlYWJmZTk5YTU5MzM3NmNkYWI4Y2U4Y2YmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Ieakk_ttMsNygVzpZZqGs1507j2GC-rqHSYH9iQQ71Q)

See [Comment](https://github.com/volcengine/verl/pull/2398#issuecomment-3157142856) for more details.

Note: AgentLoop does not directly record num_tool_calls, but records num_turns. In our scenario, you can calculate the number of tool calls by num_tool_calls = num_turns / 2 - 1.

## References and Acknowledgements

- [DeepEyes Paper](https://arxiv.org/abs/2505.14362)
- [DeepEyes Official Implementation](https://github.com/Visual-Agent/DeepEyes)

---
If you need further details for reproduction or encounter any issues, feel free to open an issue or contact the maintainers. 