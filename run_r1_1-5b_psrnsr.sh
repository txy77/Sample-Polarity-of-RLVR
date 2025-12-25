set -x

export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTHONPATH=/input/jiulin.jl/code/verl_ring_0613:PYTHONPATH
export TENSORBOARD_DIR=/home/admin/logs/tfevent

export PATH_TO_VERL="psr_nsr_a_nequal"  # Path to verl repository
export PRETRAIN_MODEL="Distill-qwen-1_5B"  # Pretrain model used for actor/ref/critic
export TRAIN_DATA="verl_train_data/HARP/random_20.parquet"
export EVAL_DATA="verl_eval_data/aime-2024.parquet"
export SAVE_ROOT_PATH="results_0708/model"
export ROLLOUT_DATA_PATH="results_0708/rollout"

# export advantage="positive"   # PSR
export advantage="negative"   # NSR
# export advantage="weighted"   # W-REINFORCE
# export positive_advantage_weight=0.1   # For W-REINFORCE only

# export RAY_DEBUG_POST_MORTEM=1

cd $PATH_TO_VERL

nohup python3 -m verl.trainer.main_ppo \
      data.train_files=${TRAIN_DATA} \
      data.val_files=${EVAL_DATA} \
      data.prompt_key=prompt \
      data.truncation=left \
      data.max_prompt_length=1024 \
      data.max_response_length=8192 \
      data.train_batch_size=32 \
      data.filter_overlong_prompts=True \
      algorithm.adv_estimator=psr_nsr \
      algorithm.advantage=${advantage} \
      algorithm.use_kl_in_reward=False \
      algorithm.kl_ctrl.kl_coef=0.0 \
      actor_rollout_ref.model.path=${PRETRAIN_MODEL} \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.model.trust_remote_code=True \
      actor_rollout_ref.actor.optim.lr=5e-5 \
      actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
      actor_rollout_ref.actor.ppo_mini_batch_size=8 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.actor.kl_loss_coef=0.0 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.fsdp_config.param_offload=True \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
      actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model] \
      actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
      actor_rollout_ref.rollout.n=8 \
      actor_rollout_ref.rollout.enable_chunked_prefill=True \
      actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
      actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.free_cache_engine=False \
      actor_rollout_ref.rollout.temperature=1.0 \
      actor_rollout_ref.rollout.top_p=1.0 \
      actor_rollout_ref.rollout.top_k=-1 \
      actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
      actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
      actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
      trainer.logger=['console'] \
      trainer.val_before_train=False \
      trainer.n_gpus_per_node=8 \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=10 \
      trainer.total_epochs=15 \
      > train_output_nsr.log 2>&1 &
      # algorithm.positive_advantage_weight=$positive_advantage_weight \