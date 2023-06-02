# DCS-ViT
## Requirements
`torch>=1.7`

`torchvision`

`pyyaml`

`huggingface_hub`

`safetensors>=0.2`

`timm`

## Train Searched dcs_mobilevit_s
`./distributed_train.sh 4 /data_dir/imagenet/ --model dcs_mobilevit_s \`

`-b 64 --workers 8 --pin-mem \`

`--sched cosine --epochs 305 --min-lr 0.0001 --lr 0.001 --warmup-lr 0.0001 --warmup-epochs 5 \`

`--opt adamw --weight-decay 0.01 \`

`--model-ema --model-ema-decay 0.9995 \`

`--aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp \`

`--smoothing 0.1 \`

`--output /output_dir`
