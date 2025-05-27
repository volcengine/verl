git config --global user.email "ligang@orby.ai"
git config --global user.name "Gang Li"

# Accelerate package download speed
sed -i 's|mirrors.tuna.tsinghua.edu.cn|archive.ubuntu.com|g' /etc/apt/sources.list

apt update
apt install -y emacs
apt install -y awscli
# urllib3<2 required by awscli
pip install 'urllib3<2'
pip install parquet-tools

# Download model.
python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-7B-Instruct')"

# Install verl lib: https://verl.readthedocs.io/en/latest/start/install.html
pip3 install -e .[vllm]

# Download and convert action description dev set
# mkdir -p ~/data/action_description/raw/
# aws s3 cp s3://orby-osu-va/mds_datasets/Q42024_Intake_Format/ActIO-ActionDescription/parquet/dev.parquet ~/data/action_description/raw/dev.parquet
# python orby/convert_action_description.py --input_file=~/data/action_description/raw/dev.parquet --split=train
# python orby/convert_action_description.py --input_file=~/data/action_description/raw/dev.parquet --split=test
