ckpt=${MTPATH}/mt.ckpts/zh-en/ckpt.exp.static.use_en/best.pt
dataset=${MTPATH}/zh-en
test_pt=${ckpt##*/}
outfile=test.out.static.${test_pt%%_dev*}_1

echo "Work on Dev"

python3 -u work.py --load_path ${ckpt} \
       --test_data ${dataset}/dev.mem.zhen.txt \
       --src_vocab_path ${dataset}/zh.vocab \
       --tgt_vocab_path ${dataset}/en.vocab \
       --output_path ${dataset}/dev.out.static.${test_pt%%_dev*} \
       --comp_bleu


echo "Work on Test"
python3 -u work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.mem.zhen.txt \
       --src_vocab_path ${dataset}/zh.vocab \
       --tgt_vocab_path ${dataset}/en.vocab \
       --output_path ${dataset}/${outfile}

spm_decode --model=${dataset}/raw/eng.model --input_format=piece < ${dataset}/${outfile} > ${dataset}/${outfile}.txt
python3 txt_xml.py --txt ${dataset}/${outfile}.txt \
        --xml ${MTPATH}/test/testsrc.xml 

