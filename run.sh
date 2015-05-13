#! /bin/bash -e

while getopts "d:r:i:l:" opt
do
    case "$opt" in
        d) data=$OPTARG;;
        r) rank=$OPTARG;;
		i) numIterations=$OPTARG;;
		l) lambdaProduct=$OPTARG;;
    esac
done

bin/spark-submit --class org.apache.spark.examples.mllib.MovieLensALS --master yarn-client --executor-memory 5000m --num-executors 40 \
/home/spark/wei/spark/examples/target/scala-2.10/spark-examples-*.jar \
--rank ${rank} --numIterations ${numIterations} --userConstraint SMOOTH --lambdaUser 0.065 --productConstraint SPARSE --lambdaProduct ${lambdaProduct} --delimiter " " \
--productNamesFile wei/mf/idvocab/idvocab.${data}.txt --kryo wei/mf/${data} | \
tee /home/spark/wei/logs/${data}/${data}rank${rank}sparse${lambdaProduct}iter${numIterations}log.txt

