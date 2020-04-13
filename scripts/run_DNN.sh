xval=5
RF='RF01851'
nepochs=50
# run under all conditions 
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task ZP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task RP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLZP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLRP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLRP -arch 1000 1000
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task FMLM1 -arch 5 5

xval=5
RF='RF00005'
nepochs=50
# run under all conditions 
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task ZP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task RP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLZP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLRP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLRP -arch 1000 1000
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task FMLM1 -arch 5 5

xval=5
RF='RF00001'
nepochs=50
# run under all conditions 
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task ZP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task RP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLZP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLRP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLRP -arch 1000 1000
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task FMLM1 -arch 5 5

xval=5
RF='RF00009'
nepochs=50
# run under all conditions 
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task ZP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task RP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLZP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLRP -arch 5 5
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task NUCSHFLRP -arch 1000 1000
python scripts/TrainDNN.py -xval $xval -epoch $nepochs -target $RF -task FMLM1 -arch 5 5
