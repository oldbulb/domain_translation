dataset=${MTPATH}/TM_NMT_data/zh-en
mkdir -p ${MTPATH}/mt.ckpts/zh-en/ckpt.vanilla

python3 -u train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --test_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt ${MTPATH}/mt.ckpts/zh-en/ckpt.vanilla \
        --total_train_steps 1000000 \
        --world_size 3 \
        --gpus 3 \
        --arch vanilla \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096