#!/bin/bash

OUTPUT_DIR=src_v3_2/outputs/rams_roberta-large_transition_new-transamr/output_lw0.05/2023-01-18-01_18_26_127631/model-epoch_41-batch_38472-f#f#dev_0.5067584347148878
log_file=${OUTPUT_DIR}/evaluate_results.log
golden_dev=data/rams/dev.jsonlines
golden_test=data/rams/test.jsonlines

rm ${log_file}

echo $OUTPUT_DIR  >> ${log_file}

echo 'generate head golden/prediction from span golden/prediction for dev'  >> ${log_file}
python scorer/transfer_results_rams.py \
--infile_golden ${golden_dev} \
--infile_prediction ${OUTPUT_DIR}/validation_predictions_span.jsonlines \
--outfile_golden ${OUTPUT_DIR}/validation_golden_head.jsonlines \
--outfile_prediction ${OUTPUT_DIR}/validation_predictions_head.jsonlines \
>> ${log_file}

echo 'generate head golden/prediction from span golden/prediction for test'  >> ${log_file}
python scorer/transfer_results_rams.py \
--infile_golden ${golden_test} \
--infile_prediction ${OUTPUT_DIR}/test_predictions_span.jsonlines \
--outfile_golden ${OUTPUT_DIR}/test_golden_head.jsonlines \
--outfile_prediction ${OUTPUT_DIR}/test_predictions_head.jsonlines \
>> ${log_file}

echo '==========================='  >> ${log_file}
echo '========for span F1========'  >> ${log_file}
echo '==========================='  >> ${log_file}

echo '==========================='  >> ${log_file}
echo 'dev'  >> ${log_file}
echo '==========================='  >> ${log_file}
python scorer/scorer.py \
--gold_file ${golden_dev} \
--pred_file ${OUTPUT_DIR}/validation_predictions_span.jsonlines \
--metrics \
>> ${log_file}

echo '==========================='  >> ${log_file}
echo 'test'  >> ${log_file}
echo '==========================='  >> ${log_file}
python scorer/scorer.py \
--gold_file ${golden_test} \
--pred_file ${OUTPUT_DIR}/test_predictions_span.jsonlines \
--metrics \
>> ${log_file}

echo '==========================='  >> ${log_file}
echo '========for head F1========'  >> ${log_file}
echo '==========================='  >> ${log_file}

echo '==========================='  >> ${log_file}
echo 'dev'  >> ${log_file}
echo '==========================='  >> ${log_file}
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/validation_golden_head.jsonlines \
--pred_file ${OUTPUT_DIR}/validation_predictions_head.jsonlines \
--metrics \
>> ${log_file}

echo '==========================='  >> ${log_file}
echo 'test'  >> ${log_file}
echo '==========================='  >> ${log_file}
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/test_golden_head.jsonlines \
--pred_file ${OUTPUT_DIR}/test_predictions_head.jsonlines \
--metrics \
>> ${log_file}

echo '==========================='  >> ${log_file}
echo 'test'  >> ${log_file}
echo '==========================='  >> ${log_file}
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/test_golden_span_identification.jsonlines \
--pred_file ${OUTPUT_DIR}/test_predictions_span_identification.jsonlines \
--metrics \
>> ${log_file}

cat ${log_file}
