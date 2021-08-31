
# pip install sentencepiece

dataset=${MTPATH}/TM_NMT_data

echo 'Train SentencePiece model on EN'
spm_train --input=${dataset}/raw_data/en.txt \
          --model_prefix=eng \
          --vocab_size=32000 \
          --character_coverage=1.0 \
          --model_type=bpe


echo 'Train SentencePiece model on ZH'
spm_train --input=${dataset}/raw_data/zh.txt \
          --model_prefix=chn \
          --vocab_size=32000 \
          --character_coverage=0.9995 \
          --model_type=bpe

echo 'Encode raw EN text into sentence pieces'
spm_encode --model=eng.model --output_format=piece < ${dataset}/raw_data/en.txt > ${dataset}/en.bpe
spm_encode --model=eng.model --output_format=piece < ${dataset}/raw_data/devref.txt > ${dataset}/dev_en.bpe

echo 'Encode raw ZH text into sentence pieces'
spm_encode --model=chn.model --output_format=piece < ${dataset}/raw_data/zh.txt > ${dataset}/zh.bpe
spm_encode --model=chn.model --output_format=piece < ${dataset}/raw_data/devsrc.txt > ${dataset}/dev_zh.bpe
spm_encode --model=chn.model --output_format=piece < ${dataset}/raw_data/testsrc.txt > ${dataset}/test_zh.bpe


mv eng* ${dataset}
mv chn* ${dataset} 
