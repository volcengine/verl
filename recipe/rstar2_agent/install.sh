SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") &>/dev/null && pwd -P)

cd $SCRIPT_DIR

# install code judge
sudo apt-get update -y && sudo apt-get install redis -y
git clone https://github.com/0xWJ/code-judge
pip install -r code-judge/requirements.txt
pip install -e code-judge

# install rstar2_agent requirements
pip install -r requirements.txt
