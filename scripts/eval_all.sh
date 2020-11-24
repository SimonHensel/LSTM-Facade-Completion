echo "Evaluating MD_LSTM NEW LOSS..."

python3.6 source/evaluate.py --model_type MD_LSTM_DISTANCE \
--checkpoint_path checkpoints/checkpoint_test/model.cpk  --hidden_size 256 \
| tee evaluation/evaled_mdlstm_newloss.txt

echo "Evaluating MDMD_LSTM NEW LOSS..."

python3.6 source/evaluate.py --model_type MDMD_LSTM_DISTANCE \
--checkpoint_path checkpoints/checkpoint_test2/model.ckpt  --hidden_size 256 \
| tee evaluation/evaled_mdmdlstm_newloss.txt

#echo "Evaluating GRID LSTM SNAKE..."

#python3.6 source/evaluate.py --model_type SNAKE_GRID_LSTM \
#--checkpoint_path checkpoints/checkpoint_gridsnake --hidden_size 256 \
#| tee evaluation/evaled_gridsnake.txt

#echo "Evaluating MD GRID LSTM SNAKE..."

#python3.6 source/evaluate.py --model_type MD_SNAKE_GRID_LSTM \
#--checkpoint_path checkpoints/checkpoint_mdgridsnake --hidden_size 128 \
#| tee evaluation/evaled_mdgridsnake.txt

#echo "Evaluating MD GRID LSTM SNAKE with ghosts..."

#python3.6 source/evaluate.py --model_type MD_SNAKE_GRID_LSTM \
#--checkpoint_path checkpoints/checkpoint_mdgridsnake_ghost --hidden_size 256 \
#| tee evaluation/evaled_mdgridsnake_ghost.txt

#echo "Evaluating MDMD_LSTM..."

#python3.6 source/evaluate.py --model_type MDMD_LSTM \
#--checkpoint_path checkpoints/checkpoint_mdmdlstm  --hidden_size 128 \
#| tee evaluation/evaled_mdmdlstm.txt
