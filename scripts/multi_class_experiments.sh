
python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task ZP -arch 5 5
python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task RP -arch 5 5
python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task NUCSHFLZP -arch 5 5
python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task NUCSHFLRP -arch 5 5

python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task ZP -arch 100 100
python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task RP -arch 100 100
python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task NUCSHFLZP -arch 100 100
python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task NUCSHFLRP -arch 100 100

python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task ZP -arch 1000 1000
python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task RP -arch 1000 1000
python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task NUCSHFLZP -arch 1000 1000
python scripts/TrainDNN_multiclass.py -xval 5 -epoch 50 -classification MUL -task NUCSHFLRP -arch 1000 1000