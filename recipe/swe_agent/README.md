### Install
1. install swe_agent
```bash
git clone https://code.byted.org/seed/SWE-Agent.git
cd SWE-agent
pip install -e .
```
or
```bash
pip install git+https://github.com/SWE-agent/SWE-agent.git
```

2. dependency
```bash
pip install -r requirements.txt
```
### Prepare Dataset
```bash
python recipe/swe_agent/dataset.py
```

### Evaluation
```
bash recipe/swe_agent/run_qwen2.5-VL-3B_dapo.sh
```