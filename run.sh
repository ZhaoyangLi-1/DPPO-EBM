for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=1 python script/run.py --config-dir=cfg/gym/pretrain/hopper-medium-v2 --config-name ft_sac_diffusion_mlp \
    train.n_train_itr=1000 train.n_steps=500 train.batch_size=256 train.replay_ratio=1 \
    train.val_freq=10 seed=$s;
  CUDA_VISIBLE_DEVICES=1 python script/run.py --config-dir=cfg/gym/pretrain/hopper-medium-v2 --config-name ft_sac_diffusion_ebm_mlp \
    model.use_ebm_reward=True model.ebm_reward_mode=dense \
    train.n_train_itr=1000 train.n_steps=500 train.batch_size=256 train.replay_ratio=1 \
    train.val_freq=10 seed=$s;
done