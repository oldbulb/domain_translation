dataset=${MTPATH}/TM_NMT_data
mkdir -p ${dataset}/zh-en 

python3 -u prepare.py --train_data_src ${dataset}/zh.bpe \
                      --train_data_tgt ${dataset}/en.bpe \
                      --vocab_src ${dataset}/zh-en/src.vocab \
                      --vocab_tgt ${dataset}/zh-en/tgt.vocab \
                      --max_len 250 \
                      --ratio 2.0 \
                      --output_file ${dataset}/zh-en/train.txt

paste -d '\t' ${dataset}/dev_zh.bpe ${dataset}/dev_en.bpe > ${dataset}/zh-en/dev.txt
cp ${dataset}/test_zh.bpe ${dataset}/zh-en/test.txt  
cp ${dataset}/en.bpe ${dataset}/zh-en/train.tgt.txt