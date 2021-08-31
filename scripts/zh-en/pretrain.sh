dataset=${MTPATH}/zh-en/
mkdir -p ${MTPATH}/mt.ckpts/zh-en/ckpt.exp.pretrain_en


python3 -u pretrain.py --train_data ${dataset}/train.enzh.txt \
        --dev_data ${dataset}/dev.enzh.txt \
        --src_vocab ${dataset}/en.vocab \
        --tgt_vocab ${dataset}/zh.vocab \
        --ckpt ${MTPATH}/mt.ckpts/zh-en/ckpt.exp.pretrain_zh \
        --world_size 1 \
        --gpus 1 \
        --dev_batch_size 64 \
        --total_train_steps 500000 \
        --layers 3 \
        --per_gpu_train_batch_size 128 \
        --bow


python3 -u pretrain.py --train_data ${dataset}/train.zhen.txt \
        --dev_data ${dataset}/dev.zhen.txt \
        --src_vocab ${dataset}/zh.vocab \
        --tgt_vocab ${dataset}/en.vocab \
        --ckpt ${MTPATH}/mt.ckpts/zh-en/ckpt.exp.pretrain_zh \
        --world_size 1 \
        --gpus 1 \
        --dev_batch_size 64 \
        --total_train_steps 500000 \
        --layers 3 \
        --per_gpu_train_batch_size 128 \
        --bow
