package org.apache.spark.examples.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.mllib.recommendation._
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.{Seconds, StreamingContext}

import org.apache.spark.examples.mllib.MovieLensALS.Params

import breeze.optimize.proximal.Constraint
import scopt.OptionParser

import scala.collection.mutable

object StreamingMovieLensSGD {

  def main(args: Array[String]) {

    val defaultParams = Params()

    val userConstraints = Constraint.values.toList.mkString(",")
    val productConstraints = Constraint.values.toList.mkString(",")

    val parser = new OptionParser[Params]("MovieLensALS") {
      head("MovieLensALS: an example app for ALS on MovieLens data.")
      opt[Int]("rank")
        .text(s"rank, default: ${defaultParams.rank}}")
        .action((x, c) => c.copy(rank = x))
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[String]("userConstraint")
        .text(s"user constraint for quadratic minimization, options ${userConstraints} default: SMOOTH")
        .action((x, c) => c.copy(userConstraint = x))
      opt[String]("productConstraint")
        .text(s"product constraint for quadratic minimization, options ${productConstraints} default: SMOOTH")
        .action((x, c) => c.copy(productConstraint = x))
      opt[Double]("lambdaUser")
        .text(s"lambda for user regularization, default: ${defaultParams.userLambda}")
        .action((x, c) => c.copy(userLambda = x))
      opt[Double]("lambdaProduct")
        .text(s"lambda for product regularization, default: ${defaultParams.productLambda}")
        .action((x, c) => c.copy(productLambda = x))
      opt[String]("delimiter")
        .text(s"sparse dataset delimiter, default: ${defaultParams.delimiter}")
        .action((x, c) => c.copy(delimiter = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Int]("numUserBlocks")
        .text(s"number of user blocks, default: ${defaultParams.numUserBlocks} (auto)")
        .action((x, c) => c.copy(numUserBlocks = x))
      opt[Int]("numProductBlocks")
        .text(s"number of product blocks, default: ${defaultParams.numProductBlocks} (auto)")
        .action((x, c) => c.copy(numProductBlocks = x))
      opt[Unit]("implicitPrefs")
        .text("use implicit preference")
        .action((_, c) => c.copy(implicitPrefs = true))
      opt[Unit]("autoParams")
        .text("use cross validation to choose params automatically")
        .action((_, c) => c.copy(autoParams = true))
      opt[String]("productNamesFile")
        .text("input paths to a MovieLens dataset of movie id to movie name")
        .action((x, c) => c.copy(productNamesFile = x))
      opt[Unit]("testStreaming")
        .text("test Streaming codes")
        .action((_, c) => c.copy(testStreaming = true))
      opt[Double]("sgdStepSize")
        .text(s"StepSize for sgd, default: ${defaultParams.sgdStepSize}")
        .action((x, c) => c.copy(sgdStepSize = x))
      opt[Int]("sgdNumIterations")
        .text(s"number of iterations for sgd, default: ${defaultParams.sgdNumIterations}")
        .action((x, c) => c.copy(sgdNumIterations = x))
      opt[Double]("sgdRegParam")
        .text(s"RegParam for sgd, default: ${defaultParams.sgdRegParam}")
        .action((x, c) => c.copy(sgdRegParam = x))
      opt[Int]("sgdBlockSize")
        .text(s"number of entrys per col and row in the block, default: ${defaultParams.sgdBlockSize}")
        .action((x, c) => c.copy(sgdBlockSize = x))
      arg[String]("<input>")
        .required()
        .text("input paths to a MovieLens dataset of ratings")
        .action((x, c) => c.copy(input = x))
      note(
        """
          |  For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class org.apache.spark.examples.mllib.StreamingMovieLensSGD \
          |  examples/target/scala-*/spark-examples-*.jar \
          |  --rank 5 --numIterations 20 --userConstraint SMOOTH --lambdaUser 0.065 --productConstraint SMOOTH --lambdaProduct 0.1 --kryo\
          |  data/mllib/sample_movielens_data.txt
          |
          |or
          |
          |  bin/run-example org.apache.spark.examples.mllib.StreamingMovieLensSGD \
          |  --rank 5 --numIterations 20 --userConstraint SMOOTH --lambdaUser 0.065 --productConstraint SMOOTH --lambdaProduct 0.1 --kryo\
          |  data/mllib/sample_movielens_data.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    // TODO: 如果分布式测试的话，不要hard code这个parallelism
    val conf = new SparkConf().set("spark.default.parallelism", "4")
      .setAppName(s"StreamingMovieLensSGD with $params")
      //.set("spark.driver.allowMultipleContexts", "true")
    if (params.kryo) {
      conf.registerKryoClasses(Array(classOf[mutable.BitSet], classOf[Rating]))
        .set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    println("===sc.defaultParallelism====================")
    println(sc.defaultParallelism)

    Logger.getRootLogger.setLevel(Level.WARN)

    val delimiter = params.delimiter

    //0 将data切成三份儿data-batch-train, data-streaming, data-test
    val ratings = sc.textFile(params.input).map { line =>
      val fields = line.split(delimiter)
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }.cache()
    val splits = ratings.randomSplit(Array(0.05, 0.75, 0.2), 1L)
    val batchTrainingData = splits(0).cache()
    val sgdTrainingData = splits(2).cache()
    val testData = splits(1).cache()
    ratings.unpersist(blocking = false)

    val numTraining = batchTrainingData.count()
    val numStream = sgdTrainingData.count()
    val numTest = testData.count()
    println(s"batchTrainingData: $numTraining, trainingSGDData: $numStream, testData: $numTest.")

    val alsModel = trainWithALS(params, batchTrainingData)
    val batchModel = new MatrixFactorizationModel(alsModel.rank, alsModel.userFeatures, alsModel.productFeatures, numTraining)
    val batchRmse = MovieLensALS.computeRmse(batchModel, testData, params.implicitPrefs)
    println(s"Test RMSE of Just Batch ALS Training= $batchRmse. " + batchModel.numEntries + " is used")

    // 2 load model1，基于这个在data-streaming上做streaming SGD train, save model2
    // 注意处理没见过的user和product
    if(params.testStreaming) {
      println("*********** Training with Streaming SGD ***********")
      val streamModel = trainWithStreamingSGD(params, sgdTrainingData, batchModel)
      val streamRmse = MovieLensALS.computeRmse(streamModel, testData, params.implicitPrefs)
      println("*********** Model with Streaming SGD ***********")
      //printModel(streamModel,"streamSGDModel")
      println(s"Test RMSE of Streaming SGD Training= $streamRmse." + streamModel.numEntries + " is used")
    } else {
      println("*********** Training with Batch SGD ***********")
      val batchSGDModel = trainWithBatchSGD(params, sgdTrainingData, batchModel)
      val batchSGDRmse = MovieLensALS.computeRmse(batchSGDModel, testData, params.implicitPrefs)
      println("*********** Model with Batch SGD ***********")
      //printModel(batchSGDModel,"batchSGDModel")
      println(s"Test RMSE of Batch SGD Training= $batchSGDRmse. " + batchSGDModel.numEntries + " is used")
    }
    println("*********** Model with Just Batch ALS ***********")
    //printModel(batchModel,"batchALSModel")
    println(s"Test RMSE of Just Batch ALS Training= $batchRmse. " + batchModel.numEntries + " is used")
    println(s"trainingData for als: $numTraining, trainingData for sgd: $numStream, testData: $numTest.")
  }

  def trainWithALS(params: Params, batchTrainingData: RDD[Rating]) =  {
    //1 用data-batch-train做ALS batch train, save model1
    // 要保证test和streaming的data在这一步是没见过的
    val userConstraint = Constraint.withName("SMOOTH")
    val productConstraint = Constraint.withName("SMOOTH")
    println(params)
    val als = new ALS()
      .setRank(params.rank)
      .setIterations(params.numIterations)
      .setUserConstraint(userConstraint)
      .setProductConstraint(productConstraint)
      .setUserLambda(params.userLambda)
      .setProductLambda(params.productLambda)
      .setImplicitPrefs(params.implicitPrefs)
      .setUserBlocks(params.numUserBlocks)
      .setProductBlocks(params.numProductBlocks)

    println(s"Quadratic minimization userConstraint ${userConstraint} productConstraint ${productConstraint}")

    //TODO: 使用原als example中的参数
    als.run(batchTrainingData)
  }

  def trainWithBatchSGD(params: Params, sgdTrainingData: RDD[Rating], batchModel: MatrixFactorizationModel) = {
    val batchSGDDataFile = sgdTrainingData.map{
      case Rating(i, j, ratings) => MatrixEntry(i, j, ratings)
    }
    //TODO:由于我们并没有做采样，因此numIterations的意思就是把整个数据集过多少遍
    new MatrixFactorizationWithSGD()
      .setStepSize(params.sgdStepSize)
      .setNumIterations(params.sgdNumIterations)
      .setRegParam(params.sgdRegParam)
      .loadInitialWeights(batchModel)
      .train(batchSGDDataFile, params.sgdBlockSize, params.sgdBlockSize)
  }

  def trainWithUnitTest(params: Params, sc: SparkContext) = {
    val initratings = List(
      Rating(1,1,1),
      Rating(1,2,1),
      Rating(2,1,3),
      Rating(2,3,3),
      Rating(3,2,5),
      Rating(3,3,4),
      Rating(3,4,1))
    val inituser = List(
      (1, Array(-0.7, 0.8)),
      (2, Array(0.5, -0.6)),
      (3, Array(0.9, -0.8)))
    val initprod = List(
      (1, Array(0.9, -0.2)),
      (2, Array(0.5, -0.3)),
      (3, Array(0.2, -0.1)),
      (4, Array(-0.5, 0.7)))
    /*
    (0,((((0,0),4 x 5 CSCMatrix
(1,1) 1.0
(2,1) 3.0
(1,2) 1.0
(3,2) 5.0
(2,3) 3.0
(3,3) 4.0
(3,4) 1.0),Some(0.0   0.0
-0.7  0.8
0.5   -0.6
0.9   -0.8  )),Some(0.0   0.0
0.9   -0.2
0.5   -0.3
0.2   -0.1
-0.5  0.7   )))
     */
    val rank = 2
    val sgdTrainingData: RDD[Rating] = sc.parallelize(initratings)
    val userFeatures: RDD[(Int, Array[Double])] = sc.parallelize(inituser)
    val productFeatures: RDD[(Int, Array[Double])] = sc.parallelize(initprod)
    val model: MatrixFactorizationModel = new  MatrixFactorizationModel(rank, userFeatures, productFeatures)
    val params = Params(sgdNumIterations = 3, sgdStepSize = 0.05, sgdRegParam = 1)

    val unitTestModel =  trainWithBatchSGD(params, sgdTrainingData, model)
    println("=======trainWithUnitTest Model===========================")
    printFeatures(unitTestModel.userFeatures)
    printFeatures(unitTestModel.productFeatures)


    val ratings: RDD[MatrixEntry] = sgdTrainingData.map{
      case Rating(i, j, ratings) => MatrixEntry(i, j, ratings)
    }
    val rowsPerBlock: Int = 1024
    val colsPerBlock: Int = 1024
    val intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK
    val finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK
  }

  def printFeatures(features:RDD[(Int, Array[Double])]) = {
    features.collect.foreach {
      case (i, arr) =>
        println(i)
        arr.foreach(x => print(x + "\t"))
        println
    }
  }

  def printModel(model: MatrixFactorizationModel, log: String = "") = {
    println("================================================")
    println(s"*******Model for $log****************************")
    println("================================================")
    println("======userFeatures=====================")
    printFeatures(model.userFeatures)
    println("======productFeatures=====================")
    printFeatures(model.productFeatures)
  }

  def trainWithStreamingSGD(params: Params, sgdTrainingData: RDD[Rating], batchModel: MatrixFactorizationModel) = {
    println("&&&&&&&&&&&&&&&& enter trainWithStreamingSGD &&&&&&&&&&&&&&&&&&&&&&&&&&&")
    // TODO: 不要hard code这个 batchDuration
    val ssc = new StreamingContext(sgdTrainingData.sparkContext, Seconds(5))
    val streamDataFile = params.input + ".stream"
    val delimiter = params.delimiter
    sgdTrainingData.map{
      case Rating(i, j, ratings) => i + delimiter + j + delimiter + ratings
    }.saveAsTextFile(streamDataFile + ".set")
    println("Streaming data dir:")
    println(streamDataFile)
    val trainingStreamData = ssc.textFileStream(streamDataFile).map { line =>
      val fields = line.split(delimiter)
      MatrixEntry(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }
    println("===trainingStreamData.foreachRDD(_.foreach(println))======================")
    trainingStreamData.foreachRDD(_.foreach(println))
    println("===trainingStreamData.print()======================")
    trainingStreamData.print()

    //TODO： 有必要做下CV选择参数stepSize和lambda（取RMSE最高的就好），在params中加入这几项
    val streamingAlgorithm = new StreamingMatrixFactorizationWithSGD()
      .setStepSize(params.sgdStepSize)
      .setNumIterations(params.sgdNumIterations)
      .setRegParam(params.sgdRegParam)
      .loadInitialWeights(batchModel)

    streamingAlgorithm.run(trainingStreamData, params.sgdBlockSize, params.sgdBlockSize)
    ssc.start()
    ssc.awaitTermination()

    // 3 比较model1和model2的效果（策略是遇到没见过的user就算平均），然后
    streamingAlgorithm.latestModel()
  }
}
