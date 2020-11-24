echo "Evaluating MD GRID LSTM SNAKE..."

python3 source/evaluate.py --model_type MD_SNAKE_GRID_LSTM \
--checkpoint_path checkpoints/checkpoint_mdgridsnake --hidden_size 128 \
| tee evaluation/evaled_mdgridsnake2.txt

echo "Evaluating MD GRID LSTM SNAKE with ghosts..."

python3 source/evaluate.py --model_type MD_SNAKE_GRID_LSTM \
--checkpoint_path checkpoints/checkpoint_mdgridsnake_ghost --hidden_size 256 \
| tee evaluation/evaled_mdgridsnake_ghost2.txt
