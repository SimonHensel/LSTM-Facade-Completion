reset
# checkpoint_mdmdlstm checkpoint_mdqrnncombi2  checkpoint_snakegrid checkpoint_mdlstm  checkpoint_mdqrnncombi  checkpoint_qrnn


echo "Evaluating MDMD_LSTM 25..."
#3125
python source/evaluate.py --model_type MDMD_LSTM \
--checkpoint_path checkpoints/checkpoint_mdmdlstm/model.ckpt  --hidden_size 2500 \
| tee evaluation/evaled_mdmdlstm_25.txt

#echo "Evaluating MD_LSTM 25 ..."
#3125
#python source/evaluate.py --model_type MD_LSTM \
#--checkpoint_path checkpoints/checkpoint_mdlstm/model.ckpt  --hidden_size 5000 \
#| tee evaluation/evaled_mdlstm25FR.txt

#echo "Evaluating MDMD_LSTM 25..."
#3125
#python source/evaluate.py --model_type MDMD_LSTM \
#--checkpoint_path checkpoints/checkpoint_mdmdlstm25_2directions/model.ckpt  --hidden_size 3125 \
#| tee evaluation/evaled_mdmdlstm_25.txt

#echo "Evaluating MD_LSTM 25..."

#python source/evaluate.py --model_type MD_LSTM \
#--checkpoint_path checkpoints/checkpoint_mdlstm25/model.ckpt  --hidden_size 4096 \
#| tee evaluation/evaled_mdlstm_25.txt

#echo "Evaluating GRID LSTM SNAKE 25..."

#python source/evaluate.py --model_type SNAKE_GRID_LSTM \
#--checkpoint_path checkpoints/checkpoint_grid25/model.ckpt --hidden_size 4096 \
#| tee evaluation/evaled_gridsnake_25.txt

#echo "Evaluating QRNN 25..."

#python source/evaluate.py --model_type QRNN \
#--checkpoint_path checkpoints/checkpoint_qrnn/model.ckpt  --hidden_size 2500 \
#| tee evaluation/evaled_qrnn_25.txt

#echo "Evaluating MD QRNN COMBI 25..."

#python source/evaluate.py --model_type MD_QRNN_COMBI \
#--checkpoint_path checkpoints/checkpoint_mdqrnncombi/model.ckpt --hidden_size 1250 \
#| tee evaluation/evaled_mdqrnncombi_25.txt

#echo "Evaluating MD QRNN COMBI2 25..."

#python source/evaluate.py --model_type MD_QRNN_COMBI2 \
#--checkpoint_path checkpoints/checkpoint_mdqrnncombi2/model.ckpt --hidden_size 1250 \
#| tee evaluation/evaled_mdqrnncombi2_25.txt
