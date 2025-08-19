# source .py_c128/bin/activate
# source /etc/network_turbo
# unset http_proxy && unset https_proxy

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

# python 安装
'''
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "verl[trl, vllm]"
pip install flash-attn --no-build-isolation
pip install bitsandbytes
pip install wandb
'''

    cmd = [
        "python", "-m", "verl.trainer.main_ppo",
        "data.train_files=$HOME/data/gsm8k/train.parquet",
        "data.val_files=$HOME/data/gsm8k/test.parquet",
        "data.train_batch_size=256",
        "data.max_prompt_length=512",
        "data.max_response_length=256",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "critic.optim.lr=1e-5",
        "critic.model.path=Qwen/Qwen2.5-0.5B-Instruct",
        "critic.ppo_micro_batch_size_per_gpu=4",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "trainer.logger=console",
        "trainer.val_before_train=False",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=10",
        "trainer.test_freq=10",
        "trainer.total_epochs=15",
    ]