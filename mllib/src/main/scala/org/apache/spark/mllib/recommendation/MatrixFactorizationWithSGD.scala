package org.apache.spark.mllib.recommendation

import breeze.linalg.{DenseMatrix => BrzMatrix, DenseVector => BrzVector, shuffle}
import org.apache.spark.{HashPartitioner, Logging}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRowMatrix, CoordinateMatrix, IndexedRow, MatrixEntry}
import org.apache.spark.mllib.linalg.{DenseVector, Matrices, Matrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel


class MatrixFactorizationWithSGD  private[mllib] (
  private var numUserBlocks: Int,
  private var numProductBlocks: Int,
  private var stepSize: Double, //initial step size for the first step
  private var numIterations: Int, //number of iterations that SGD would run through the whole data set
  private var regParam: Double) //L2 regularization parameter
  extends Logging with Serializable {

  def this() = this(-1, -1, 0.01, 30, 0.1)

  /** Set the step size for gradient descent. Default: 0.1. */
  def setStepSize(stepSize: Double): this.type = {
    this.stepSize = stepSize
    this
  }

  /** Set the number of iterations of gradient descent to run per update. Default: 50. */
  def setNumIterations(numIterations: Int): this.type = {
    this.numIterations = numIterations
    this
  }

  /** Set the fraction of each batch to use for updates. Default: 1.0. */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the number of blocks for both user blocks and product blocks to parallelize the computation
   * into; pass -1 for an auto-configured number of blocks. Default: -1.
   */
  def setBlocks(numBlocks: Int): this.type = {
    this.numUserBlocks = numBlocks
    this.numProductBlocks = numBlocks
    this
  }

  /**
   * Set the number of user blocks to parallelize the computation.
   */
  def setUserBlocks(numUserBlocks: Int): this.type = {
    this.numUserBlocks = numUserBlocks
    this
  }

  /**
   * Set the number of product blocks to parallelize the computation.
   */
  def setProductBlocks(numProductBlocks: Int): this.type = {
    this.numProductBlocks = numProductBlocks
    this
  }

  /** Set the initial weights. Default: [0.0, 0.0]. */
  def loadInitialWeights(matrixFactorizationModel: MatrixFactorizationModel): this.type = {
    this.model = matrixFactorizationModel
    this
  }

  /** The model to be updated and used for prediction. */
  protected var model: MatrixFactorizationModel = null

  /**
   * Partitioner used by SGD.
   */
  private[recommendation] type SGDPartitioner = org.apache.spark.HashPartitioner

  def getPartitioner(ratings: RDD[MatrixEntry]) = {
    val sc = ratings.sparkContext
    val numUserBlocks = if (this.numUserBlocks == -1) {
      println("sc.defaultParallelism: " + sc.defaultParallelism + ", ratings.partitions.size: "+ ratings.partitions.size)
      math.max(sc.defaultParallelism, ratings.partitions.size / 2)
    } else {
      this.numUserBlocks
    }
    val numProductBlocks = if (this.numProductBlocks == -1) {
      math.max(sc.defaultParallelism, ratings.partitions.size / 2)
    } else {
      this.numProductBlocks
    }
    val userPartitioner = new SGDPartitioner(numUserBlocks)
    val productPartitioner = new SGDPartitioner(numProductBlocks)
    (userPartitioner, productPartitioner)
  }

  def newModel(rank: Int,
               blockUserFeatures: RDD[(Int, BrzMatrix[Double])],
               blockProductFeatures: RDD[(Int, BrzMatrix[Double])],
               numEntries: Long,
               rowsPerBlock: Int,
               colsPerBlock: Int,
               RDDStorageLevel: StorageLevel): MatrixFactorizationModel = {
    // 这里加HashPartitioner但并不需要跟其他RDD协作，只是为了增加并行度加速预测
    val numRowBlocks = blockUserFeatures.count.toInt
    val numColBlocks = blockProductFeatures.count.toInt
    val modelUserFeatures = new BlockMatrix(blockUserFeatures.map(x => ((x._1, 0), Matrices.fromBreeze(x._2))), rowsPerBlock, rank)
      .toIndexedRowMatrix().rows.map{
      case IndexedRow(i, features) =>
        (i.toInt, features.toArray)
    }.partitionBy(new HashPartitioner(numRowBlocks)).setName("modelUserFeatures").persist(RDDStorageLevel)
    val modelProductFeatures = new BlockMatrix(blockProductFeatures.map(x => ((x._1, 0), Matrices.fromBreeze(x._2))), colsPerBlock, rank)
      .toIndexedRowMatrix().rows.map{
      case IndexedRow(i, features) =>
        (i.toInt, features.toArray)
    }.partitionBy(new HashPartitioner(numColBlocks)).setName("modelProductFeatures").persist(RDDStorageLevel)
    new MatrixFactorizationModel(rank, modelUserFeatures, modelProductFeatures, numEntries)
  }

  def fillFeatures(features: RDD[(Int, Array[Double])], dataKeys: RDD[Int], rank: Int): RDD[(Int, Array[Double])] = {
    /*
    // Fill features with avg values
    val avgValue = features.aggregate[BrzVector[Double]](BrzVector.zeros(rank)) (
      (U, T) => (U + new BrzVector(T._2)),
      (U1, U2) => (U1 + U2)
    )
    avgValue /= features.count.toDouble
    val avgArray = avgValue.toArray
    */
    val avgArray = BrzVector.zeros[Double](rank).toArray

    println("init Value for missing user/products")
    avgArray.foreach(x => print(x + " "))
    println

    val avgFeatures = dataKeys.subtract(features.keys).map{x=>(x, avgArray.clone)}
    println("avgFeatures.count:" + avgFeatures.count)
    avgFeatures.foreach(x => print(x._1 + ", "))
    println
    features.union(avgFeatures)
  }

  /**
   * Implementation of the ALS algorithm.
   * Return: (MatrixFactorizationModel, mergedMatrixFactorizationModel)
   */
  def train(
      ratings: RDD[MatrixEntry],
      rowsPerBlock: Int = 1024,
      colsPerBlock: Int = 1024,
      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): (MatrixFactorizationModel, MatrixFactorizationModel) = {
    val sc = ratings.sparkContext
    val rank = model.rank
    val userFeatures = model.userFeatures
    val productFeatures = model.productFeatures
    val historyNum = model.numEntries

    val (userPartitioner, productPartitioner) = getPartitioner(ratings)

    val currentNum = ratings.count
    var sofarNum = historyNum
    val historyRate = historyNum.toDouble / (historyNum + currentNum)
    val currentRate = 1 - historyRate
    println(s"historyNum: $historyNum  currentNum: $currentNum")
    println(s"historyRate: $historyRate  currentRate: $currentRate")
    println("userFeatures.partitioner:" + userFeatures.partitioner)
    println("productFeatures.partitioner:" + productFeatures.partitioner)
    /*
    val oldTrainingRMSE = MatrixFactorizationWithSGD.computeRmseMatrixEntry(model, ratings)
    println("****************************************************************************************")
    println(s"====Google training RMSE SGD for old data in training set RMSE: $oldTrainingRMSE")
    println("****************************************************************************************")
    */

    println("################ Adding Init Feature Values for new users ##########################")
    val filledUserFeatures = fillFeatures(userFeatures, ratings.keyBy(x => x.i.toInt).keys.distinct, rank)
    println("################ Adding Init Feature Values for new productes ##########################")
    val filledProductFeatures = fillFeatures(productFeatures, ratings.keyBy(x => x.j.toInt).keys.distinct, rank)
/*
    // Don't fill the features with init values
    val filledUserFeatures = userFeatures
    val filledProductFeatures = productFeatures
*/

    //TODO: 其实blockRatings，blockUserFeatures和blockProductFeatures本身还是需要分区，不然的话参数通信还是很大...
    //TODO: 根据blockRatings.numRowBlocks和blockRatings.numRows调整blockRatings.rowsPerBlock的大小
    val blockRatings:BlockMatrix = new CoordinateMatrix(ratings).toBlockMatrix(rowsPerBlock, colsPerBlock)
    val blockRatingKeyByUser: RDD[(Int, ((Int, Int), Matrix))] = blockRatings.blocks.keyBy{case ((x, y), matrix) => x}
      .partitionBy(userPartitioner).setName("blockRatingKeyByUser").persist(intermediateRDDStorageLevel)

    def cutFeatureToBlocks(features: RDD[(Int, Array[Double])], numPerBlock: Int): RDD[(Int, BrzMatrix[Double])]= {
      new IndexedRowMatrix(features.map(x =>
        IndexedRow(x._1.toLong, new DenseVector(x._2)))).toBlockMatrix(numPerBlock, rank).blocks.map {
        case ((x, y), featureMatrix) => (x, featureMatrix.toBreeze.toDenseMatrix)
      }
    }
    var blockUserFeatures = cutFeatureToBlocks(filledUserFeatures, rowsPerBlock)
      .partitionBy(userPartitioner).setName("blockUserFeatures").persist(intermediateRDDStorageLevel)
    var blockProductFeatures = cutFeatureToBlocks(filledProductFeatures, colsPerBlock)
      .partitionBy(productPartitioner).setName("blockProductFeatures").persist(intermediateRDDStorageLevel)

    println("blockRatingKeyByUser.partitioner: "+blockRatingKeyByUser.partitioner)
    println("blockUserFeatures.partitioner: "+blockUserFeatures.partitioner)
    println("blockProductFeatures.partitioner: "+blockProductFeatures.partitioner)

    /* Activate persist and checkpoint */
    blockUserFeatures.checkpoint()
    blockProductFeatures.checkpoint()
    blockUserFeatures.count
    blockProductFeatures.count

    // TODO: 这里其实是对init model中缺失值进行填充后的features
    val initBlockUserFeatures = blockUserFeatures
    val initBlockProductFeatures = blockProductFeatures

    val tmpZeroModel = newModel(rank, initBlockUserFeatures, initBlockProductFeatures, sofarNum,
      rowsPerBlock, colsPerBlock, intermediateRDDStorageLevel)
    val iterZeroRMSE = MatrixFactorizationWithSGD.computeRmseMatrixEntry(tmpZeroModel, ratings)
    tmpZeroModel.userFeatures.unpersist()
    tmpZeroModel.productFeatures.unpersist()
    println("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&Z&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    println(s"====Google training RMSE SGD in iter 0 RMSE: $iterZeroRMSE=====")
    println("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*")

    /* Acitive blockRatings.cache() */
    println("==BlockMatrix==================")
    println("ratings.count:"+ ratings.count)
    println("blockRatings.blocks.count:" + blockRatings.blocks.count)
    println("===blockRatings:Row and Col num of blocks=======")
    println(blockRatings.numRowBlocks)
    println(blockRatings.numColBlocks)
    println("===blockRatings:Row and Col num of entrys=======")
    println(blockRatings.numRows)
    println(blockRatings.numCols)

    //TODO(wli12): 不需要miniBatchFraction 这里就是每次一个元素的sgd，更快，虽然收敛性有待探讨
    //TODO:通过join处理没出现过的new user，这个点还需要测试
    val stratumNum = math.max(blockRatings.numColBlocks, blockRatings.numRowBlocks)
    println(s"stratumNum: $stratumNum")
    for (iter <- 1 to numIterations) {
      for (stratumID <- 0 until stratumNum) {
        blockUserFeatures.setName("blockUserFeatures").persist(intermediateRDDStorageLevel)
        blockProductFeatures.setName("blockProductFeatures").persist(intermediateRDDStorageLevel)
        val previousBlockUserFeatures = blockUserFeatures
        val previousBlockProductFeatures = blockProductFeatures

        val newFactors = computeFactors(blockRatingKeyByUser, blockUserFeatures, blockProductFeatures,
          iter, stratumID, stratumNum, rowsPerBlock, colsPerBlock, userPartitioner, productPartitioner, rank, sofarNum)
        blockUserFeatures = newFactors._1
        blockProductFeatures = newFactors._2
        sofarNum = newFactors._3

        previousBlockUserFeatures.unpersist()
        previousBlockProductFeatures.unpersist()
      }
      if (sc.checkpointDir.isDefined && (iter % 3 == 0)) {
        blockUserFeatures.checkpoint()
        blockProductFeatures.checkpoint()
        blockUserFeatures.count
        blockProductFeatures.count
      }
      val tmpModel = newModel(rank, blockUserFeatures, blockProductFeatures, sofarNum,
        rowsPerBlock, colsPerBlock, intermediateRDDStorageLevel)
      val iterRMSE = MatrixFactorizationWithSGD.computeRmseMatrixEntry(tmpModel, ratings)
      tmpModel.userFeatures.unpersist()
      tmpModel.productFeatures.unpersist()
      println("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
      println(s"===================Google training RMSE SGD in iter $iter RMSE: $iterRMSE====================")
      println("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*")
    }

    def getMergedBlockFeatures(features: RDD[(Int, BrzMatrix[Double])], initFeatures: RDD[(Int, BrzMatrix[Double])]) = {
      features.join(initFeatures).map {
        case (i, (features, oldFeatures)) =>
          (i, historyRate * features + currentRate * oldFeatures)
      }
    }
    val mergedBlockUserFeatures = getMergedBlockFeatures(blockUserFeatures, initBlockUserFeatures)
    val mergedBlockProductFeatures = getMergedBlockFeatures(blockProductFeatures, initBlockProductFeatures)
    mergedBlockUserFeatures.count
    mergedBlockProductFeatures.count

    blockRatingKeyByUser.unpersist()
    initBlockUserFeatures.unpersist()
    initBlockProductFeatures.unpersist()

    (newModel(rank, blockUserFeatures, blockProductFeatures, historyNum + currentNum, rowsPerBlock, colsPerBlock, finalRDDStorageLevel),
      newModel(rank, mergedBlockUserFeatures, mergedBlockProductFeatures, historyNum + currentNum, rowsPerBlock, colsPerBlock, finalRDDStorageLevel))
  }

  def computeFactors(
      blockRatingKeyByUser: RDD[(Int, ((Int, Int), Matrix))],
      blockUserFeatures: RDD[(Int, BrzMatrix[Double])],
      blockProductFeatures: RDD[(Int, BrzMatrix[Double])],
      iter: Int,
      stratumID: Int,
      stratumNum: Int,
      rowsPerBlock: Int,
      colsPerBlock: Int,
      userPartitioner: SGDPartitioner,
      productPartitioner: SGDPartitioner,
      rank: Int,
      historyNum: Long,
      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) : (RDD[(Int, BrzMatrix[Double])], RDD[(Int, BrzMatrix[Double])], Long) = {
    //TODO:检验join操作是否一一对应
    val accum = blockUserFeatures.sparkContext.accumulator(0)

    println(s"********Yeah! iter:$iter stratumID:$stratumID *********************")

    /* Active cache*/
    println("********blockRatings keys********")
    blockRatingKeyByUser.foreach{case (k,((m, n), _)) => println(m, n)}
    println("********blockUserFeatures keys********")
    blockUserFeatures.foreach(x=>println(x._1))
    println("********blockProductFeatures keys********")
    blockProductFeatures.foreach(x=>println(x._1))

    println("blockRatingKeyByUser.partitioner: "+blockRatingKeyByUser.partitioner)
    println("blockUserFeatures.partitioner: "+blockUserFeatures.partitioner)
    println("blockProductFeatures.partitioner: "+blockProductFeatures.partitioner)

    val stratumWithFeatures = blockRatingKeyByUser.filter {
      case (k, ((x, y), matrix)) =>
        if ((x + y) % stratumNum == stratumID) true else false
    }.leftOuterJoin(blockUserFeatures).map {
      case (k, (((x, y), matrix), userFeature)) =>
        (y, (((x, y), matrix), userFeature))
    }.leftOuterJoin(blockProductFeatures).map {
      case (k, ((((x, y), matrix), userFeature), productFeature)) =>
        (x, ((((x, y), matrix), userFeature), productFeature))
    }.partitionBy(userPartitioner).setName("stratumWithFeatures").persist(intermediateRDDStorageLevel)

    println("********stratumWithFeatures keys********")
    stratumWithFeatures.foreach {
      case (k, ((((m, n), matrix), optionUserFeatureBlock), optionProductFeatureBlock)) => println(m, n)
    }

    val updatedFeatures: RDD[(Int,(Int, (Boolean, BrzMatrix[Double])))] = stratumWithFeatures.flatMapValues {
      case ((((m, n), matrix), optionUserFeatureBlock), optionProductFeatureBlock) =>
        //TODO: 处理并没有FeatureBlock或者FeatureBlock缺行
        def getFeatureMatrixTrans(optionFeatureBlock: Option[BrzMatrix[Double]], numPerBlock: Int):BrzMatrix[Double] = {
          (optionFeatureBlock match {
            //TODO: 要看看toBreeze以后，行列是否一致
            case Some(featureBlock) => featureBlock
            case None =>
              println("====================ERROR rand initFeatureBlock=================================")
              BrzMatrix.rand[Double](numPerBlock, rank)
          }).t
        }
        val userFeatureMatrixTrans = getFeatureMatrixTrans(optionUserFeatureBlock, rowsPerBlock)
        val ProductFeatureMatrixTrans = getFeatureMatrixTrans(optionProductFeatureBlock, colsPerBlock)
        val matrixEntrys = matrix.toBreeze.activeIterator.toArray

        /*
        val beforeRMSE = MatrixFactorizationWithSGD.localRmseZeroUnseen(userFeatureMatrixTrans, ProductFeatureMatrixTrans, matrixEntrys)
        println("*******************************************************************************")
        println(s"=======Google training RMSE before iter $iter computation: $beforeRMSE========")
        println("*******************************************************************************")
        */

        println("********************************************")
        println("=======SGD UPDATE shuffled ITEMS============")
        println("********************************************")

        val shuffleIndices = shuffle(BrzVector.range(0, matrixEntrys.length))
        shuffleIndices.toArray.zipWithIndex.foreach { x =>
          val ((i, j), rating) = matrixEntrys(x._1)
          val index = x._2
          val predictionError = rating - userFeatureMatrixTrans(::, i).t * ProductFeatureMatrixTrans(::, j)

          // TODO: 加sqrt会使stepsize比较大，有比较大的机会把不好的局部解冲掉
          //val thisIterStepSize = stepSize / math.sqrt(historyNum + index + 1)
          //val thisIterStepSize = stepSize / iter*iter
          //val thisIterStepSize = stepSize / iter
          val thisIterStepSize = stepSize / math.sqrt(iter)
          //val thisIterStepSize = stepSize / math.pow(iter, 0.25)

          userFeatureMatrixTrans(::, i) :+= thisIterStepSize * (predictionError * ProductFeatureMatrixTrans(::, j) - regParam * userFeatureMatrixTrans(::, i))
          ProductFeatureMatrixTrans(::, j) :+= thisIterStepSize * (predictionError * userFeatureMatrixTrans(::, i) - regParam * ProductFeatureMatrixTrans(::, j))

          val predictionError2 = rating - userFeatureMatrixTrans(::, i).t * ProductFeatureMatrixTrans(::, j)
          val improve = predictionError - predictionError2
          //println(s"index:$index step:$thisIterStepSize" + '\n' + s"improve:$improve")
        }

        /*
        val afterRMSE = MatrixFactorizationWithSGD.localRmseZeroUnseen(userFeatureMatrixTrans, ProductFeatureMatrixTrans, matrixEntrys)
        println("****************************************************************************")
        println(s"=======Google training RMSE after iter $iter computation: $afterRMSE=======")
        println("****************************************************************************")
        */

        accum += matrixEntrys.length
        List((m, (true, userFeatureMatrixTrans.t)), (n, (false, ProductFeatureMatrixTrans.t)))
    }.setName("updatedFeatures").persist(intermediateRDDStorageLevel)

    val updatedBlockUserFeatures = updatedFeatures.filter(_._2._2._1).mapValues(x => x._2._2)
      .setName("updatedBlockUserFeatures").persist(intermediateRDDStorageLevel)
    val updatedBlockProductFeatures = updatedFeatures.filter(!_._2._2._1).map {case (_, x) =>(x._1, x._2._2)}
      .partitionBy(productPartitioner).setName("updatedBlockProductFeatures").persist(intermediateRDDStorageLevel)

    val newBlockUserFeatures = blockUserFeatures.subtractByKey(updatedBlockUserFeatures).union(updatedBlockUserFeatures)
      .setName("newBlockUserFeatures").persist(intermediateRDDStorageLevel)
    val newBlockProductFeatures = blockProductFeatures.subtractByKey(updatedBlockProductFeatures).union(updatedBlockProductFeatures)
      .setName("newBlockProductFeatures").persist(intermediateRDDStorageLevel)

    /* Activate persist of newBlockUserFeatures and newBlockProductFeatures*/
    println("newBlockUserFeatures.count:" + newBlockUserFeatures.count())
    println("newBlockProductFeatures.count:" + newBlockProductFeatures.count())

    stratumWithFeatures.unpersist()
    updatedFeatures.unpersist()
    updatedBlockUserFeatures.unpersist()
    updatedBlockProductFeatures.unpersist()

    println(s"====Google accum.value:"+accum.value+"==========================")

    (newBlockUserFeatures, newBlockProductFeatures, historyNum + accum.value)
  }
}

object MatrixFactorizationWithSGD {
  def mapPredictedRating(r: Double, implicitPrefs: Boolean) = {
    if (implicitPrefs) math.max(math.min(r, 1.0), 0.0)
    else r
  }

  //有mean这个action
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], implicitPrefs: Boolean) = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map { x =>
      ((x.user, x.product), mapPredictedRating(x.rating, implicitPrefs))
    }.join(data.map(x => ((x.user, x.product), x.rating))).values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }

  def computeRmseMatrixEntry(model: MatrixFactorizationModel, data: RDD[MatrixEntry], implicitPrefs: Boolean = false) = {
    computeRmse(model, data.map{case MatrixEntry(i,j,r) => Rating(i.toInt,j.toInt,r)}, implicitPrefs)
  }

  /*
   * Predict the unseen data as 0
   */
  def localRmseZeroUnseen(userFeatureMatrixTrans: BrzMatrix[Double],ProductFeatureMatrixTrans: BrzMatrix[Double], data: Array[((Int, Int), Double)]) = {
    val rmse = data.map{
      case ((i, j), rating) =>
        val predictionError = rating - userFeatureMatrixTrans(::, i).t * ProductFeatureMatrixTrans(::, j)
        predictionError * predictionError
    }.reduce(_ + _)
    Math.sqrt(rmse / data.length)
  }

  /*
   * Predict the unseen data as 0, using local datas: Array[((Int, Int), Double)]
   */
  // TODO:这里有training data大小的collect 不应该用在分布式里头
  def rddRmseZeroUnseen(model: MatrixFactorizationModel, data: RDD[MatrixEntry]) = {
    val userFeatures = model.userFeatures.map {
      case (i, features) => IndexedRow(i.toLong, new DenseVector(features))
    }
    val productFeatures = model.productFeatures.map {
      case (i, features) => IndexedRow(i.toLong, new DenseVector(features))
    }
    val datas = data.map{
      case MatrixEntry(i,j,k) => ((i.toInt,j.toInt),k)
    }.collect
    localRmseZeroUnseen(new IndexedRowMatrix(userFeatures).toBreeze.t, new IndexedRowMatrix(productFeatures).toBreeze.t, datas)
  }

  /*
   * Jump the unseen data, using local userFeatures: Map[Int, Array[Double]]
   */
  // TODO:这里有userFeatures+productFeatures大小的collect 不应该用在分布式里头
  def rddRmseJumpUnseen(model: MatrixFactorizationModel, data: RDD[MatrixEntry]) = {
    val userFeatures = model.userFeatures.collect.toMap
    val productFeatures = model.productFeatures.collect.toMap

    println("userFeatures")
    println(userFeatures.keys.toArray.sorted)
    println("max" + userFeatures.keys.toArray.sorted.max)
    println("userFeatures.keys.size:" + userFeatures.keys.size)
    println("productFeatures")
    println("max" + productFeatures.keys.toArray.sorted.max)
    println("productFeatures.keys.size:" + productFeatures.keys.size)

    val count = data.sparkContext.accumulator(0)
    val rmse = data.flatMap{
      case MatrixEntry(i,j,rating) =>
        if(userFeatures.contains(i.toInt) && productFeatures.contains(j.toInt)){
          val predictionError = (rating - new DenseVector(userFeatures(i.toInt)).toBreeze.t * new DenseVector(productFeatures(j.toInt)).toBreeze)
          count += 1
          List(predictionError * predictionError)
        } else List()
    }.reduce(_+_)
    Math.sqrt(rmse / count.value)
  }
}
