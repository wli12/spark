#! /bin/bash -e
if [[ $# < 1 ]]; then
  echo "Usage: ./run_local.sh -r rank -l lambdaProduct -i numIterations"
  exit 0
fi

while getopts "r:i:l:" opt
do
    case "$opt" in
        r) rank=$OPTARG;;
		i) numIterations=$OPTARG;;
		l) lambdaProduct=$OPTARG;;
    esac
done

echo "bin/run-example org.apache.spark.examples.mllib.MovieLensALS \
--rank ${rank} --numIterations ${numIterations} --userConstraint SMOOTH --lambdaUser 0.065 --productConstraint SPARSE --lambdaProduct ${lambdaProduct} --delimiter " " \
--productNamesFile ../../data/idvocab.kos.txt  ../../data/tfidf.kos.txt/part-00000 |\
tee ../../data/tfidf.kos.txt/rank${rank}sparse${lambdaProduct}iter${numIterations}log.txt"

bin/run-example org.apache.spark.examples.mllib.MovieLensALS \
--rank ${rank} --numIterations ${numIterations} --userConstraint SMOOTH --lambdaUser 0.065 --productConstraint SPARSE --lambdaProduct ${lambdaProduct} --delimiter " " \
--productNamesFile ../../data/idvocab.kos.txt  ../../data/tfidf.kos.txt/part-00000 |\
tee ../../data/tfidf.kos.txt/rank${rank}sparse${lambdaProduct}iter${numIterations}log.txt

echo "Full logs: ../../data/tfidf.kos.txt/rank${rank}sparse${lambdaProduct}iter${numIterations}log.txt"