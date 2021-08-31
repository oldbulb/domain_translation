set -e
  
ckpt_prefix=${MTPATH}/mt.ckpts
dataset_prefix=${MTPATH}/zh-en
ckpt_folder=zh-en/ckpt.exp.pretrain_zh/epoch30_batch469999_acc0.90
data_field=raw/zh.mono.sort

python3 build_index.py \
        --input_file ${dataset_prefix}/${dataset}/${data_field} \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/${dataset}/zh.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --max_training_instances 20000000 \
        --batch_size 512

python3 build_index.py \
        --input_file ${dataset_prefix}/${dataset}/${data_field} \
        --ckpt_path ${ckpt_prefix}/${ckpt_folder}/response_encoder \
        --args_path ${ckpt_prefix}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/${dataset}/zh.vocab \
        --index_path ${ckpt_prefix}/${ckpt_folder}/mips_index \
        --max_training_instances 20000000 \
        --batch_size 512 \
        --only_dump_feat

cp ${dataset_prefix}/${dataset}/${data_field} ${ckpt_prefix}/${ckpt_folder}/candidates.txt
cp -r ${ckpt_prefix}/${ckpt_folder} ${ckpt_prefix}/${ckpt_folder}_zh

