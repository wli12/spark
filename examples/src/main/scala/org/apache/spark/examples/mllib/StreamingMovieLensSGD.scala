package org.apache.spark.examples.mllib

import breeze.optimize.proximal.Constraint
import org.apache.log4j.{Level, Logger}
import org.apache.spark.examples.mllib.MovieLensALS.Params
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.HashPartitioner
import scopt.OptionParser

import scala.collection.mutable
import java.util.Calendar
import java.text.SimpleDateFormat

object StreamingMovieLensSGD {

  def main(args: Array[String]) {

    val defaultParams = Params()

    val userConstraints = Constraint.values.toList.mkString(",")
    val productConstraints = Constraint.values.toList.mkString(",")

    val parser = new OptionParser[Params]("StreamingMovieLensSGD") {
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
      opt[Int]("rowsPerBlock")
        .text(s"number of entrys per col and row in the block, default: ${defaultParams.rowsPerBlock}")
        .action((x, c) => c.copy(rowsPerBlock = x))
      opt[Int]("colsPerBlock")
        .text(s"number of col and row of blocks, default: ${defaultParams.colsPerBlock}")
        .action((x, c) => c.copy(colsPerBlock = x))
      opt[Double]("sgdDataRate")
        .text(s"Rate of data for sgd, 0<=rate<=0.8 default: ${defaultParams.sgdDataRate}")
        .action((x, c) => c.copy(sgdDataRate = x))
      opt[Unit]("timeStamp")
        .text("Partition the data by timestamp")
        .action((_, c) => c.copy(timeStamp = true))
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
    val conf = new SparkConf()
      .setAppName(s"StreamingMovieLensSGD with $params")
    if (params.kryo) {
      conf.registerKryoClasses(Array(classOf[mutable.BitSet], classOf[Rating]))
        .set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    println("===Let's Go====================")
    printTime()
    println("===sc.defaultParallelism====================")
    println(sc.defaultParallelism)
    if(sc.defaultParallelism < 8) {
      sc.conf.set("spark.default.parallelism", "8")
    }
    println(sc.defaultParallelism)
    println("checkpointDir:" + sc.checkpointDir)
    sc.setCheckpointDir("spark-checkpoint-dir")
    println("checkpointDir:" + sc.checkpointDir)

    Logger.getRootLogger.setLevel(Level.WARN)

    println("=======trainWithUnitTest BEGIN====================================")
    trainWithUnitTest(sc)
    println("=======trainWithUnitTest END====================================")

    val delimiter = params.delimiter

    val sgdDataRate = params.sgdDataRate
    assert(sgdDataRate <= 0.8 && sgdDataRate >= 0)

    //0 将data切成三份儿data-batch-train, data-streaming, data-test
    def getSplits(input: String):(RDD[Rating], RDD[Rating], RDD[Rating]) = {
      val ratings = sc.textFile(input).map { line =>
        val fields = line.split(delimiter)
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
      }
      val splits = ratings.randomSplit(Array(0.8 - sgdDataRate, sgdDataRate, 0.2), 1L)
      (splits(0), splits(1), splits(2))
    }

    val (alsTrainingData, sgdTrainingData, testData) = getSplits(params.input)
    alsTrainingData.setName("alsTrainingData").persist(StorageLevel.MEMORY_AND_DISK)
    sgdTrainingData.setName("sgdTrainingData").persist(StorageLevel.MEMORY_AND_DISK)
    testData.setName("testData").persist(StorageLevel.MEMORY_AND_DISK)
    // Activate persist
    val numALSTrain = alsTrainingData.count()
    val numSGD = sgdTrainingData.count()
    val numTest = testData.count()
    println(s"alsTrainingData: $numALSTrain, trainingSGDData: $numSGD, testData: $numTest.")

    // TODO: 这里的baseline仅仅用来看看优化效果，由于get了流式数据，不符合实验设定，RMSE比实际更好
    val (baselineRmse, avgbaselineRmse) = baseline(alsTrainingData.union(sgdTrainingData), testData)

    println(s"Training " + 100 * (0.8 - sgdDataRate) + "% data with ALS frist")
    val alsModel = trainWithALS(params, alsTrainingData)
    val ALSbatchModel = new MatrixFactorizationModel(alsModel.rank, alsModel.userFeatures, alsModel.productFeatures, numALSTrain)
    ALSbatchModel.userFeatures.checkpoint()
    ALSbatchModel.productFeatures.checkpoint()
    val ALSRmse = MovieLensALS.computeRmse(ALSbatchModel, testData, params.implicitPrefs)
    println(s"Test RMSE of Just Batch ALS Training= $ALSRmse. " + ALSbatchModel.numEntries + " is used")

    val ALSbatchimprovement = (baselineRmse - ALSRmse) / baselineRmse * 100
    println("The Batch ALS model improves the baseline by " + "%1.2f".format(ALSbatchimprovement) + "%.")
    val avgALSbatchimprovement = (avgbaselineRmse - ALSRmse) / avgbaselineRmse * 100
    println("The Batch ALS model improves the avg baseline by " + "%1.2f".format(avgALSbatchimprovement) + "%.")

    // 2 load model1，基于这个在data-streaming上做streaming SGD train, save model2
    // 注意处理没见过的user和product
    println(s"Training " + 100 * sgdDataRate + "% data with SGD after batch ALS")
    if(params.testStreaming) {
      println("*********** Training with Streaming SGD ***********")
      val streamModel = trainWithStreamingSGD(params, sgdTrainingData, ALSbatchModel)
      val streamRmse = MovieLensALS.computeRmse(streamModel, testData, params.implicitPrefs)
      println("*********** Model with Streaming SGD ***********")
      //printModel(streamModel,"streamSGDModel")
      println(s"Test RMSE of Streaming SGD Training= $streamRmse." + streamModel.numEntries + " is used")
    } else {
      println("*********** Training with Batch SGD ***********")
      val (batchSGDModel, mergedbatchSGDModel) = trainWithBatchSGD(params, sgdTrainingData, ALSbatchModel)

      val batchSGDRmse = MovieLensALS.computeRmse(batchSGDModel, testData, params.implicitPrefs)
      println("*********** Model with Batch SGD ***********")
      //printModel(batchSGDModel,"batchSGDModel")
      println(s"Test RMSE of Batch SGD Training= $batchSGDRmse. " + batchSGDModel.numEntries + " is used")
      val batchSGDimprovement = (baselineRmse - batchSGDRmse) / baselineRmse * 100
      println("The Batch SGD model improves the baseline by " + "%1.2f".format(batchSGDimprovement) + "%.")
      val avgbatchSGDimprovement = (avgbaselineRmse - batchSGDRmse) / avgbaselineRmse * 100
      println("The Batch SGD model improves the avg baseline by " + "%1.2f".format(avgbatchSGDimprovement) + "%.")
      val ALSbatchSGDimprovement = (ALSRmse - batchSGDRmse) / ALSRmse * 100
      println("The Batch SGD model improves the ALS by " + "%1.2f".format(ALSbatchSGDimprovement) + "%.")

      val mergedbatchSGDRmse = MovieLensALS.computeRmse(mergedbatchSGDModel, testData, params.implicitPrefs)
      println("*********** Model with merged Batch SGD ***********")
      //printModel(mergedbatchSGDModel,"mergedbatchSGDModel")
      println(s"Test RMSE of merged Batch SGD Training= $mergedbatchSGDRmse. " + mergedbatchSGDModel.numEntries + " is used")
      val mergedbatchSGDimprovement = (baselineRmse - mergedbatchSGDRmse) / baselineRmse * 100
      println("The merged Batch SGD model improves the baseline by " + "%1.2f".format(mergedbatchSGDimprovement) + "%.")
      val avgmergedbatchSGDimprovement = (avgbaselineRmse - mergedbatchSGDRmse) / avgbaselineRmse * 100
      println("The merged Batch SGD model improves the avg baseline by " + "%1.2f".format(avgmergedbatchSGDimprovement) + "%.")
      val ALSmergedbatchSGDimprovement = (ALSRmse - mergedbatchSGDRmse) / ALSRmse * 100
      println("The merged Batch SGD model improves the ALS by " + "%1.2f".format(ALSmergedbatchSGDimprovement) + "%.")

      computeRMSE(ALSbatchModel, batchSGDModel, mergedbatchSGDModel, alsTrainingData, sgdTrainingData, testData)
    }

    println("*********** Model with Just Batch ALS ***********")
    //printModel(ALSbatchModel,"batchALSModel")
    println(s"Test RMSE of Just Batch ALS Training= $ALSRmse. " + ALSbatchModel.numEntries + " is used")
    println("The Batch ALS model improves the baseline by " + "%1.2f".format(ALSbatchimprovement) + "%.")
    println("The Batch ALS model improves the avg baseline by " + "%1.2f".format(avgALSbatchimprovement) + "%.")
    println(s"trainingData for als: $numALSTrain, trainingData for sgd: $numSGD, testData: $numTest.")
    println(s"Baseline RMSE: $baselineRmse")
    println(s"AVG Baseline RMSE: $avgbaselineRmse")

    printTime()

    alsTrainingData.unpersist()
    sgdTrainingData.unpersist()
    testData.unpersist()
  }

  def printTime() = {
    val today = Calendar.getInstance.getTime
    val curTimeFormat = new SimpleDateFormat("yy-MM-dd HH:mm:ss")
    println("completed " + curTimeFormat.format(today))
  }

  // TODO: 有collect操作，长度为product数量的Map
  def baseline(trainingData: RDD[Rating], testData: RDD[Rating]) = {
    // create a naive baseline and compare it with the best model
    val meanRating: Double = trainingData.map(_.rating).mean
    val baselineRmse =
      math.sqrt(testData.map(x => (meanRating - x.rating) * (meanRating - x.rating)).mean)

    // create a baseline based on avarage rating for old product and meanRating for new product
    val avgRating: Map[Int, Double] = trainingData.map{ x => (x.product, x.rating)}.combineByKey(
      (v) => (v, 1),
      (c:(Double, Int), v) => (c._1 + v, c._2 + 1),
      (c1:(Double, Int), c2:(Double, Int)) => (c1._1 + c2._1, c1._2 + c2._2)
    ).mapValues(x => x._1/x._2).collect.toMap
    val avgbaselineRmse =
      math.sqrt(testData.map(x =>
        if (avgRating.contains(x.product)) {
          (avgRating(x.product) - x.rating) * (avgRating(x.product) - x.rating)
        } else {
          (meanRating - x.rating) * (meanRating - x.rating)
        }).mean)
    (baselineRmse, avgbaselineRmse)
  }

  //有collect，大小为User/Procuct key size的几个set
  def computeRMSE(ALSModel: MatrixFactorizationModel, SGDModel: MatrixFactorizationModel, mergedModel: MatrixFactorizationModel,
                  alsTrainingData:RDD[Rating], sgdTrainingData:RDD[Rating], testData: RDD[Rating]): List[Double] = {
    val alsUserKeys = alsTrainingData.map(x => x.user).distinct.collect.toSet
    val alsProductKeys = alsTrainingData.map(x => x.product).distinct.collect.toSet

    val sgdUserKeys = sgdTrainingData.map(x => x.user).distinct.collect.toSet ++ alsUserKeys
    val sgdProductKeys = sgdTrainingData.map(x => x.product).distinct.collect.toSet ++ alsProductKeys

    val oldTestData = testData.filter{
      x =>  alsUserKeys.contains(x.user) && alsProductKeys.contains(x.product)
    }
    val newTestData = testData.filter{
      x => !(alsUserKeys.contains(x.user) && alsProductKeys.contains(x.product)) &&
        (sgdUserKeys.contains(x.user) && sgdProductKeys.contains(x.product))
    }
    newTestData.map(x=>(x.user,x.product)).foreach(println)
    println(sgdUserKeys -- alsUserKeys)
    println(sgdProductKeys -- alsProductKeys)
    println(s"oldTestData count: "+oldTestData.count +" newTestData count: "+ newTestData.count)
    println(s"alsUserKeys count: "+alsUserKeys.size +" sgdUserKeys count: "+ sgdUserKeys.size)
    println(s"alsProductKeys count: "+alsProductKeys.size +" sgdProductKeys count: "+ sgdProductKeys.size)

    val oldalsRmse = MovieLensALS.computeRmse(ALSModel, oldTestData, false)
    val newalsRmse_invalid = MovieLensALS.computeRmse(ALSModel, newTestData, false)

    val (oldbaselineRmse, oldavgbaselineRmse) = baseline(alsTrainingData, oldTestData)
    val (newbaselineRmse, newavgbaselineRmse) = baseline(alsTrainingData, newTestData)

    val oldsgdRmse = MovieLensALS.computeRmse(SGDModel, oldTestData, false)
    val newsgdRmse = MovieLensALS.computeRmse(SGDModel, newTestData, false)

    val oldmergedRmse = MovieLensALS.computeRmse(mergedModel, oldTestData, false)
    val newmergedRmse = MovieLensALS.computeRmse(mergedModel, newTestData, false)

    println("=============================================================")
    println("************ old/new data RMSE compare ***********************")
    println("=============================================================")
    println(s"oldbaselineRmse:$oldbaselineRmse"+'\t'+s" newbaselineRmse:$newbaselineRmse, " + '\n' +
      s"oldavgbaselineRmse:$oldavgbaselineRmse"+'\t'+s" newavgbaselineRmse:$newavgbaselineRmse, " + '\n' +
      s"oldalsRmse:$oldalsRmse"+'\t'+s" newalsRmse_invalid:$newalsRmse_invalid, " + '\n' +
      s"oldsgdRmse:$oldsgdRmse"+'\t'+s" newsgdRmse:$newsgdRmse, " + '\n' +
      s"oldmergedRmse:$oldmergedRmse"+'\t'+s" newmergedRmse:$newmergedRmse")

    List(oldalsRmse, newalsRmse_invalid, newbaselineRmse, newavgbaselineRmse, oldsgdRmse, oldsgdRmse, oldmergedRmse, oldmergedRmse)
  }

  def printFeatures(features:RDD[(Int, Array[Double])]) = {
    features.foreach {
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

  def trainWithBatchSGD(params: Params, sgdTrainingData: RDD[Rating],
                        batchModel: MatrixFactorizationModel): (MatrixFactorizationModel, MatrixFactorizationModel)  = {
    val batchSGDDataFile = sgdTrainingData.map{
      case Rating(i, j, ratings) => MatrixEntry(i, j, ratings)
    }

    //TODO:由于我们并没有做采样，因此numIterations的意思就是把整个数据集过多少遍
    new MatrixFactorizationWithSGD()
      .setStepSize(params.sgdStepSize)
      .setNumIterations(params.sgdNumIterations)
      .setRegParam(params.sgdRegParam)
      .loadInitialWeights(batchModel)
      .setUserBlocks(params.numUserBlocks)
      .setProductBlocks(params.numProductBlocks)
      .train(batchSGDDataFile, params.rowsPerBlock, params.colsPerBlock)
  }

  def trainWithStreamingSGD(params: Params, sgdTrainingData: RDD[Rating],
                            batchModel: MatrixFactorizationModel): MatrixFactorizationModel = {
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
      .setUserBlocks(params.numUserBlocks)
      .setProductBlocks(params.numProductBlocks)
      .loadInitialWeights(batchModel)

    streamingAlgorithm.run(trainingStreamData, params.rowsPerBlock, params.colsPerBlock)
    ssc.start()
    ssc.awaitTermination()

    // 3 比较model1和model2的效果（策略是遇到没见过的user就算平均），然后
    streamingAlgorithm.latestModel()
  }

  def trainWithUnitTest(sc: SparkContext) = {
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

    val rank = 2
    val sgdTrainingData: RDD[Rating] = sc.parallelize(initratings)
    val userFeatures: RDD[(Int, Array[Double])] = sc.parallelize(inituser)
    val productFeatures: RDD[(Int, Array[Double])] = sc.parallelize(initprod)
    val model: MatrixFactorizationModel = new  MatrixFactorizationModel(rank, userFeatures, productFeatures)
    val params = Params(sgdNumIterations = 3, sgdStepSize = 0.05, sgdRegParam = 1)

    val (unitTestModel, mergedUnitTestModel) = trainWithBatchSGD(params, sgdTrainingData, model)
    println("=======trainWithUnitTest Model===========================")
    printFeatures(unitTestModel.userFeatures)
    printFeatures(unitTestModel.productFeatures)

    /*
    val ratings: RDD[MatrixEntry] = sgdTrainingData.map{
      case Rating(i, j, ratings) => MatrixEntry(i, j, ratings)
    }
    val rowsPerBlock: Int = 1024
    val colsPerBlock: Int = 1024
    val intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK
    val finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

    import org.apache.spark.mllib.recommendation.MatrixFactorizationWithSGD._
    */
  }
}
