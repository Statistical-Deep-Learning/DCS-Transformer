./distributed_train.sh 4 /data-drive/backup/yyang409/data/imagenet/ --model mobilevit_s \
-b 64 --workers 8 --pin-mem \
--sched cosine --epochs 305 --min-lr 0.0001 --lr 0.001 --warmup-lr 0.0001 --warmup-epochs 5 \
--opt adamw --weight-decay 0.01 \
--model-ema --model-ema-decay 0.9995 \
--aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp \
--smoothing 0.1 \
--output /home/local/ASUAD/ywan1053/ImageNet/output/mobilevit_s \
--resume /home/local/ASUAD/ywan1053/ImageNet/output/mobilevit_s/20230406-160807-mobilevit_s-256/checkpoint-40.pth.tar