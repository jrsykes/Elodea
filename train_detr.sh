cd scripts/detr/

source activate detr

python main.py \
    --epochs 300 \
    --batch_size 2 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /local/scratch/jrs596/dat/ElodeaProject/Elodea_BB \
    --output_dir /local/scratch/jrs596/dat/ElodeaProject \
    --world_size 2
    