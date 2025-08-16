# source /etc/network_turbo
# unset http_proxy && unset https_proxy
# source .py_env/bin/activate

# Github SSH
'''
ssh-keygen -t ed25519 -C "tinpingchan@yahoo.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
打开 GitHub → Settings → SSH and GPG keys → New SSH key → 粘贴上面的公钥 → 保存
ssh -T git@github.com

git remote set-url origin git@github.com:ChanTinPing/LLM_RL_basic.git
'''
import yaml
print(yaml.__version__)
