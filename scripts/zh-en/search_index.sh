set -e

dataset=${MTPATH}/zh-en
vocab=${MTPATH}/zh-en/en.vocab

ckpt_folder=${MTPATH}/mt.ckpts/zh-en/ckpt.exp.pretrain_zh/epoch30_batch469999_acc0.90_zh

python3 -u search_index.py \
        --input_file ${dataset}/train.zhen.txt \
        --output_file ${dataset}/train.mem.enzh.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 1024

python3 -u search_index.py \
        --input_file ${dataset}/dev.enzh.txt \
        --output_file ${dataset}/dev.mem.zhen.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 1024

python3 -u search_index.py \
        --input_file ${dataset}/test.enzh.txt \
        --output_file ${dataset}/test.mem.zhen.txt \
        --ckpt_path ${ckpt_folder}/query_encoder \
        --args_path ${ckpt_folder}/args \
        --vocab_path ${vocab} \
        --index_file ${ckpt_folder}/candidates.txt \
        --index_path ${ckpt_folder}/mips_index \
        --allow_hit \
        --batch_size 1024


