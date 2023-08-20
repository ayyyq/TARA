#!/bin/bash

OUTPUT_DIR=src_v3_5_aux/outputs/wikievents_roberta-large_transition/output_lw0.2/2023-01-19-14_10_16_014504
log_file=${OUTPUT_DIR}/evaluate_results.log
golden_dev=data/wikievents/transfer-dev.jsonl
golden_test=data/wikievents/transfer-test.jsonl

rm ${log_file}

#echo $OUTPUT_DIR  >> ${log_file}
#echo 'generate head/coref golden/prediction from span golden/prediction for dev'  >> ${log_file}
#python scorer/transfer_results_wikievent.py \
#--infile_golden ${golden_dev} \
#--infile_prediction ${OUTPUT_DIR}/validation_predictions_span.jsonlines \
#--outdir ${OUTPUT_DIR} \
#--split validation \
#>> ${log_file}

echo 'generate head/coref golden/prediction from span golden/prediction for test' >> ${log_file}
python scorer/transfer_results_wikievent.py \
--infile_golden ${golden_test} \
--infile_prediction ${OUTPUT_DIR}/test_predictions_span.jsonlines \
--outdir ${OUTPUT_DIR} \
--split test \
>> ${log_file}

echo '==========================================' >> ${log_file}
echo '========for Head F1 Classification========' >> ${log_file}
echo '==========================================' >> ${log_file}

#echo '===========================' >> ${log_file}
#echo 'dev' >> ${log_file}
#echo '===========================' >> ${log_file}
#python scorer/scorer.py \
#--gold_file ${OUTPUT_DIR}/validation_golden_head.jsonlines \
#--pred_file ${OUTPUT_DIR}/validation_predictions_head.jsonlines \
#--metrics \
#>> ${log_file}

echo '===========================' >> ${log_file}
echo 'test' >> ${log_file}
echo '===========================' >> ${log_file}
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/test_golden_head.jsonlines \
--pred_file ${OUTPUT_DIR}/test_predictions_head.jsonlines \
--metrics \
>> ${log_file}

echo '==========================================' >> ${log_file}
echo '========for Coref F1 Classification========' >> ${log_file}
echo '==========================================' >> ${log_file}

#echo '===========================' >> ${log_file}
#echo 'dev' >> ${log_file}
#echo '===========================' >> ${log_file}
#python scorer/scorer.py \
#--gold_file ${OUTPUT_DIR}/validation_golden_coref.jsonlines \
#--pred_file ${OUTPUT_DIR}/validation_predictions_coref.jsonlines \
#--metrics \
#>> ${log_file}

echo '===========================' >> ${log_file}
echo 'test' >> ${log_file}
echo '===========================' >> ${log_file}
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/test_golden_coref.jsonlines \
--pred_file ${OUTPUT_DIR}/test_predictions_coref.jsonlines \
--metrics \
>> ${log_file}

echo '==========================================' >> ${log_file}
echo '========for Head F1 Identification========' >> ${log_file}
echo '==========================================' >> ${log_file}

#echo '===========================' >> ${log_file}
#echo 'dev' >> ${log_file}
#echo '===========================' >> ${log_file}
#python scorer/scorer.py \
#--gold_file ${OUTPUT_DIR}/validation_golden_head_identification.jsonlines \
#--pred_file ${OUTPUT_DIR}/validation_predictions_head_identification.jsonlines \
#--metrics \
#>> ${log_file}

echo '===========================' >> ${log_file}
echo 'test' >> ${log_file}
echo '===========================' >> ${log_file}
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/test_golden_head_identification.jsonlines \
--pred_file ${OUTPUT_DIR}/test_predictions_head_identification.jsonlines \
--metrics \
>> ${log_file}

echo '==========================================' >> ${log_file}
echo '========for Coref F1 Identification========' >> ${log_file}
echo '==========================================' >> ${log_file}

#echo '===========================' >> ${log_file}
#echo 'dev' >> ${log_file}
#echo '===========================' >> ${log_file}
#python scorer/scorer.py \
#--gold_file ${OUTPUT_DIR}/validation_golden_coref_identification.jsonlines \
#--pred_file ${OUTPUT_DIR}/validation_predictions_coref_identification.jsonlines \
#--metrics \
#>> ${log_file}

echo '===========================' >> ${log_file}
echo 'test' >> ${log_file}
echo '===========================' >> ${log_file}
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/test_golden_coref_identification.jsonlines \
--pred_file ${OUTPUT_DIR}/test_predictions_coref_identification.jsonlines \
--metrics \
>> ${log_file}

cat ${log_file}
