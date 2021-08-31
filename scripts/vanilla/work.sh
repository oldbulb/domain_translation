ckpt=${MTPATH}/mt.ckpts/zh-en/ckpt.vanilla/epoch31_batch95999_devbleu62.27_testbleu61.54
dataset=${MTPATH}/TM_NMT_data/zh-en
test_pt=${ckpt##*/}

echo "Work on Dev"

python3 -u work.py --load_path ${ckpt} \
       --test_data ${dataset}/dev.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/dev.out.vanilla.${test_pt%%_dev*} \
       --comp_bleu
 
echo "Work on Test"
python3 -u work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.vanilla.${test_pt%%_dev*}

spm_decode --model=${dataset}/eng.model --input_format=piece < ${dataset}/test.out.vanilla.${test_pt%%_dev*} > ${dataset}/test.out.vanilla.${test_pt%%_dev*}.txt

python3 txt_xml.py --txt ${dataset}/test.out.vanilla.${test_pt%%_dev*}.txt \
        --xml ${MTPATH}/test/testsrc.xml 
