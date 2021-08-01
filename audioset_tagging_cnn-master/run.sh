source activate py36

python pytorch/main.py train --workspace=./ --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type=Cnn14 --loss_type=clip_bce  --balanced=balanced --augmentation=mixup --batch_size=24 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda