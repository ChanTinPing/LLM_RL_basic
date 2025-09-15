import os, sys, subprocess, argparse, yaml, random, torch, shlex, re
from datetime import datetime
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
    if not cfg['rollout_dir']:
        rollout_data_dir = 'null'
    else:
        rollout_data_dir = f"{cfg['rollout_dir']}/{datetime.now().strftime('%y%m%d_%H%M%S')}"

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        
        # ===== 数据 =====
        f"data.train_files={cfg['train_parquet']}",
        f"data.val_files={cfg['aime25_parquet']}", 
        f"data.train_batch_size={cfg['train_batch_size']}",
        f"data.max_prompt_length={cfg['max_prompt_len']}",
        f"data.max_response_length={cfg['max_resp_len']}",
        "trainer.val_before_train=false",     ##########
        "trainer.test_freq=0",                ##########

        # ===== 模型 =====
        f"actor_rollout_ref.model.path={cfg['base_model']}",   
        "actor_rollout_ref.model.trust_remote_code=true",
        "actor_rollout_ref.actor.strategy=megatron",
        "critic.strategy=megatron",
        f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={cfg['tp_size']}",
        f"actor_rollout_ref.actor.optim.lr={cfg['learning_rate']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg['ppo_micro_batch_per_gpu']}",
        f"actor_rollout_ref.actor.ppo_epochs={cfg['ppo_epochs']}",
        
        # ===== rollout =====
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg['mem_utilz']}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.load_format=dummy_megatron",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg['tp_size']}",
        f"actor_rollout_ref.rollout.dtype={cfg['dtype']}", 
        f"+trainer.rollout_data_dir={rollout_data_dir}",
        
        # ===== GRPO =====
        f"actor_rollout_ref.rollout.n={cfg['rollout_n']}",
        f"actor_rollout_ref.actor.use_kl_loss={cfg['use_kl']}",
        f"actor_rollout_ref.actor.kl_loss_coef={cfg['kl_coef']}",
        f"actor_rollout_ref.actor.kl_loss_type={cfg['kl_type']}",
        f"algorithm.adv_estimator={cfg['adv_estimator']}",
        "algorithm.kl_ctrl.kl_coef=0.001",  
        f"+is_dapo={cfg['is_dapo']}",

        # ===== trainer =====
        f"trainer.default_local_dir={cfg['output_dir']}",
        f"trainer.logger={cfg['logging']}",
        f"trainer.total_epochs={cfg['total_epochs']}",  #############
        f"trainer.n_gpus_per_node={cfg['n_gpus']}",
        f"trainer.nnodes={cfg['nnodes']}",
        f"trainer.save_freq={cfg['save_steps']}",   
        "trainer.val_before_train=true",
        f"trainer.test_freq={cfg['test_steps']}",  
        f"trainer.max_actor_ckpt_to_keep={cfg['max_actor_ckpt_to_keep']}",
        
        # ===== reward =====
        "reward_model.enable=false",
        f"custom_reward_function.path={os.path.abspath(cfg['reward_func'])}",
        "custom_reward_function.name=compute_score",
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["NVTE_DEBUG"] = "1"
    env["NVTE_DEBUG_LEVEL"] = "2"
    env["TENSORBOARD_DIR"] = f"/root/autodl-tmp/LLM_RL_basic/outputs/tb/{timestamp}"
    env["TT_ENABLE"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    
    # 把终端输出存在 logfile 里
    logfile = f"/root/autodl-tmp/LLM_RL_basic/outputs/log/driver.stdout.{timestamp}.log"

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    with open(logfile, "a", buffering=1) as f:  # 行缓冲
        # text=True + bufsize=1 => 按行读取
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in proc.stdout:
                print(line, end="")  # 终端实时显示
                clean_line = ansi_escape.sub('', line)    # 去掉颜色码
                f.write(clean_line)   
        finally:
            proc.stdout.close()
            rc = proc.wait()
    if rc != 0:
        raise SystemExit(rc)
    
if __name__ == "__main__":
    main()
