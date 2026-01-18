@echo off
for %%s in (42 1 2 3 4 5 10 20 100 999) do (
    echo Running seed %%s...
    python train_gnn.py --graph-json warehouse_graph.json --epochs 1000 --hidden-dim 64 --lr 0.05 --weight-decay 0.0001 --dropout 0.6 --patience 100 --seed %%s >> results.txt
)
echo Finished!
pause
