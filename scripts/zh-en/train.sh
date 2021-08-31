dataset=${MTPATH}/zh-en
mkdir -p ${MTPATH}/mt.ckpts/zh-en/ckpt.exp.static.use_en_3

python3 train.py --train_data ${dataset}/train.mem.zhen.txt \
        --dev_data ${dataset}/dev.mem.zhen.txt \
        --test_data ${dataset}/test.mem.zhen.txt \
        --src_vocab ${dataset}/zh.vocab \
        --tgt_vocab ${dataset}/en.vocab \
        --ckpt ${MTPATH}/mt.ckpts/zh-en/ckpt.exp.static.use_en_3 \
        --world_size 3 \
        --gpus 3 \
	--arch src_mem \
        --use_mem_score \
        --dev_batch_size 1024 \
        --per_gpu_train_batch_size 1024 \
        --total_train_steps 800000 \
        --eval_every 1000 \

