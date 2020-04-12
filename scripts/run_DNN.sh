xval=5
RF='RF00005'
nepochs=10
# RF00005 under all conditions 
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task ZP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task RP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLZP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLRP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLRP -arch 1000 1000
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task FMLM1 -arch 5 5
