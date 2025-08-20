import os, sys, subprocess, argparse, yaml, random, torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cudnn 固定模式（可能稍微慢，但结果可复现）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # python 自带的哈希种子
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_parquet(parquet_path: str):
    if os.path.exists(parquet_path):
        print(f"[ok] found parquet: {parquet_path}")
        return
    print(f"[build] parquet not found, creating -> {parquet_path}")
    cmd = ['python', os.path.join(HERE, "prepare_deepscaler_parquet.py")]
    subprocess.run(cmd, check=True, cwd=ROOT)   # 用同一解释器，cwd=ROOT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "grpo.yaml"))
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    set_seed(cfg["seed"])
    ensure_parquet(os.path.join(ROOT, cfg["train_parquet"]))

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",

        f"+actor_rollout_ref.actor.fsdp_config.model_dtype={cfg['dtype']}",
        #f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg['tp_size']}",

        # ===== 数据 =====
        f"data.train_files={os.path.abspath(cfg['train_parquet'])}",
        f"data.val_files={os.path.abspath(cfg['train_parquet'])}", 
        f"data.train_batch_size={cfg['train_batch_size']}",
        f"data.max_prompt_length={cfg['max_prompt_len']}",
        f"data.max_response_length={cfg['max_resp_len']}",
        "trainer.val_before_train=false",     ##########
        "trainer.test_freq=0",                ##########

        # ===== 模型 =====
        f"actor_rollout_ref.model.path={cfg['base_model']}",   
        "actor_rollout_ref.model.trust_remote_code=true",
        f"actor_rollout_ref.actor.optim.lr={cfg['learning_rate']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg['ppo_micro_batch_per_gpu']}",
        f"actor_rollout_ref.actor.ppo_epochs={cfg['ppo_epochs']}",
        
        # ===== rollout =====
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",

        # ===== GRPO =====
        f"actor_rollout_ref.rollout.n={cfg['rollout_n']}",
        f"actor_rollout_ref.actor.use_kl_loss={cfg['use_kl']}",
        f"actor_rollout_ref.actor.kl_loss_coef={cfg['kl_coef']}",
        f"actor_rollout_ref.actor.kl_loss_type={cfg['kl_type']}",
        f"algorithm.adv_estimator={cfg['adv_estimator']}",
        "algorithm.kl_ctrl.kl_coef=0.001",  

        # ===== trainer =====
        f"trainer.default_local_dir={cfg['output_dir']}",
        f"trainer.logger={cfg['logging']}",
        f"trainer.total_epochs={cfg['total_epochs']}",  #############
        f"trainer.n_gpus_per_node={cfg['n_gpus']}",
        f"trainer.nnodes={cfg['nnodes']}",
        f"trainer.save_freq={cfg['save_steps']}",   
        #f"trainer.test_freq={cfg['test_freq']}",   
    ]

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    subprocess.run(cmd, env=env)
    
if __name__ == "__main__":
    main()
