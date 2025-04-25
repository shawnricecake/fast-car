# Pack the evaluation results into a zip file.

cd evaluation_results

zip -r ../evaluation_results.zip .
cd ..

# [Optional] get the total score of your submission file.
python vbench_eval/cal_final_score.py --zip_file evaluation_results.zip --model_name t2v_model_tmp

rm evaluation_results.zip
rm -r t2v_model_tmp/
