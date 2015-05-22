package org.apache.spark.mllib.recommendation

import breeze.linalg.{DenseMatrix => BrzMatrix, DenseVector => BrzVector, shuffle}
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{DenseVector, Matrices}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

class MatrixFactorizationWithSGD  private[mllib] (
  private var stepSize: Double, //initial step size for the first step
  private var numIterations: Int, //number of iterations that SGD would run through the whole data set
  private var regParam: Double) //L2 regularization parameter
  extends Logging with Serializable {

  def this() = this(0.1, 50, 1.0)

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

  /** The model to be updated and used for prediction. */
  protected var model: MatrixFactorizationModel = null

  /** Set the initial weights. Default: [0.0, 0.0]. */
  def loadInitialWeights(matrixFactorizationModel: MatrixFactorizationModel): this.type = {
    this.model = matrixFactorizationModel
    this
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
    val rank = model.rank
    val userFeatures = model.userFeatures
    val productFeatures = model.productFeatures
    val historyNum = model.numEntries
    val currentNum = ratings.count
    val historyRate = historyNum.toDouble / (historyNum + currentNum)
    val currentRate = 1 - historyRate

    var sofarNum = historyNum
    val initRMSE = MatrixFactorizationWithSGD.rddRmse(model, ratings)
    val initArrayRMSE = MatrixFactorizationWithSGD.rddToArrayRmse(model, ratings)
    val initArrayRMSE2 = MatrixFactorizationWithSGD.rddToArrayRmse2(model, ratings)

    println(s"historyNum: $historyNum  currentNum: $currentNum")
    println(s"historyRate: $historyRate  currentRate: $currentRate")
    println("****************************************************************************************")
    println(s"====Google training RMSE SGD initRMSE: $initRMSE initArrayRMSE: $initArrayRMSE initArrayRMSE2:$initArrayRMSE2================================")
    println("****************************************************************************************")

    //TODO: 看看这个rank作为参数对不对
    //TODO: 根据blockRatings.numRowBlocks和blockRatings.numRows调整blockRatings.rowsPerBlock的大小
    val blockRatings:BlockMatrix = new CoordinateMatrix(ratings).toBlockMatrix(rowsPerBlock, colsPerBlock).persist(intermediateRDDStorageLevel)
    var blockUserFeatures: RDD[(Int, BrzMatrix[Double])] = new IndexedRowMatrix(userFeatures.map(x =>
      IndexedRow(x._1.toLong, new DenseVector(x._2)))).toBlockMatrix(rowsPerBlock, rank).blocks.map {
         case ((x, y), featureMatrix) => (x, featureMatrix.toBreeze.toDenseMatrix)
      }.persist(intermediateRDDStorageLevel)
    var blockProductFeatures: RDD[(Int, BrzMatrix[Double])] = new IndexedRowMatrix(productFeatures.map(x =>
      IndexedRow(x._1.toLong, new DenseVector(x._2)))).toBlockMatrix(colsPerBlock, rank).blocks.map {
         case ((x, y), featureMatrix) => (x, featureMatrix.toBreeze.toDenseMatrix)
      }.persist(intermediateRDDStorageLevel)
    val initBlockUserFeatures = blockUserFeatures
    val initBlockProductFeatures = blockProductFeatures

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
        blockUserFeatures.persist(intermediateRDDStorageLevel)
        blockProductFeatures.persist(intermediateRDDStorageLevel)
        val previousBlockUserFeatures = blockUserFeatures
        val previousBlockProductFeatures = blockProductFeatures

        val newFactors = computeFactors(blockRatings, blockUserFeatures, blockProductFeatures,
          iter, stratumID, rowsPerBlock, colsPerBlock, rank, sofarNum)
        blockUserFeatures = newFactors._1
        blockProductFeatures = newFactors._2
        sofarNum = newFactors._3

        previousBlockUserFeatures.unpersist()
        previousBlockProductFeatures.unpersist()
      }
    }

    val mergedBlockUserFeatures = blockUserFeatures.join(initBlockUserFeatures).map {
      case (i, (features, oldFeatures)) =>
        (i, historyRate * features + currentRate * oldFeatures)
    }
    val mergedBlockProductFeatures = blockProductFeatures.join(initBlockProductFeatures).map{
      case (i, (features, oldFeatures)) =>
        (i, historyRate * features + currentRate * oldFeatures)
    }

    def newModel(blockUserFeatures: RDD[(Int, BrzMatrix[Double])],
                 blockProductFeatures: RDD[(Int, BrzMatrix[Double])]): MatrixFactorizationModel = {
      val updatedUserFeatures = new BlockMatrix(blockUserFeatures.map(x => ((x._1, 0), Matrices.fromBreeze(x._2))), rowsPerBlock, rank)
        .toIndexedRowMatrix().rows.map{
        case IndexedRow(i, features) =>
          (i.toInt, features.toArray)
      }.persist(finalRDDStorageLevel)
      val updatedProductFeatures = new BlockMatrix(blockProductFeatures.map(x => ((x._1, 0), Matrices.fromBreeze(x._2))), colsPerBlock, rank)
        .toIndexedRowMatrix().rows.map{
        case IndexedRow(i, features) =>
          (i.toInt, features.toArray)
      }.persist(finalRDDStorageLevel)
      new MatrixFactorizationModel(rank, updatedUserFeatures, updatedProductFeatures, historyNum + currentNum)
    }

    val models = (newModel(blockUserFeatures, blockProductFeatures), newModel(mergedBlockUserFeatures, mergedBlockProductFeatures))
    /*
    blockRatings.blocks.unpersist()
    blockUserFeatures.unpersist()
    blockProductFeatures.unpersist()
    */
    models
  }

  def computeFactors(
                      blockRatings: BlockMatrix,
                      blockUserFeatures: RDD[(Int, BrzMatrix[Double])],
                      blockProductFeatures: RDD[(Int, BrzMatrix[Double])],
                      iter: Int,
                      stratumID: Int,
                      rowsPerBlock: Int,
                      colsPerBlock: Int,
                      rank: Int,
                      historyNum: Long,
                      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) : (RDD[(Int, BrzMatrix[Double])], RDD[(Int, BrzMatrix[Double])], Long) = {
    //TODO:检验join操作是否一一对应
    val stratumNum = math.max(blockRatings.numColBlocks, blockRatings.numRowBlocks)
    val accum = blockUserFeatures.sparkContext.accumulator(0)
    val stratumWithFeatures = blockRatings.blocks.filter {
      case ((x, y), matrix) =>
        if ((x + y) % stratumNum == stratumID) true else false
    }.map {
      case ((x, y), matrix) => (x, ((x, y), matrix))
    }.leftOuterJoin(blockUserFeatures).map {
      case (k, (((x, y), matrix), userFeature)) => (y, (((x, y), matrix), userFeature))
    }.leftOuterJoin(blockProductFeatures).persist(intermediateRDDStorageLevel)

    println(s"********Yeah! iter:$iter stratumID:$stratumID *********************")
    println("blockRating blocks" + blockRatings.blocks.count)
    println("stratumWithFeatures.count" + stratumWithFeatures.count)

    println("********blockRatings keys********")
    blockRatings.blocks.foreach{case ((m, n), _) => println(m, n)}
    println("********blockUserFeatures keys********")
    blockUserFeatures.foreach(x=>println(x._1))
    println("********blockProductFeatures keys********")
    blockProductFeatures.foreach(x=>println(x._1))
    println("********stratumWithFeatures keys********")
    stratumWithFeatures.foreach {
      case (k, ((((m, n), matrix), optionUserFeatureBlock), optionProductFeatureBlock)) => println(m, n)
    }

    val updatedFeatures: RDD[(Int, (Boolean, BrzMatrix[Double]))] = stratumWithFeatures.flatMap {
      case (k, ((((m, n), matrix), optionUserFeatureBlock), optionProductFeatureBlock)) =>
        //TODO: 处理并没有FeatureBlock或者FeatureBlock缺行
        val userFeatureMatrixTrans:BrzMatrix[Double] = (optionUserFeatureBlock match {
          //TODO: 要看看toBreeze以后，行列是否一致
          case Some(featureBlock) => featureBlock
          case None =>
            println("==================================ERROR rand userFeature===========================================")
            BrzMatrix.rand[Double](rowsPerBlock, rank)
        }).t
        val ProductFeatureMatrixTrans:BrzMatrix[Double] = (optionProductFeatureBlock match {
          case Some(featureBlock) => featureBlock
          case None =>
            println("==================================ERROR rand ProductFeature===========================================")
            BrzMatrix.rand[Double](colsPerBlock, rank)
        }).t
        val matrixEntrys = matrix.toBreeze.activeIterator.toArray

        val beforeRMSE = MatrixFactorizationWithSGD.dataArrayRmse(userFeatureMatrixTrans, ProductFeatureMatrixTrans, matrixEntrys)
        println("*******************************************************************************")
        println(s"=======Google training RMSE before iter $iter computation: $beforeRMSE========")
        println("*******************************************************************************")

        println("********************************************")
        println("=======SGD UPDATE shuffled ITEMS============")
        println("*******************************************")

        val shuffleIndices = shuffle(BrzVector.range(0, matrixEntrys.length))
        shuffleIndices.toArray.zipWithIndex.foreach { x =>
          val ((i, j), rating) = matrixEntrys(x._1)
          val index = x._2
          val predictionError = rating - userFeatureMatrixTrans(::, i).t * ProductFeatureMatrixTrans(::, j)

          // TODO: 加sqrt会使stepsize比较大，有比较大的机会把不好的局部解冲掉
          //val thisIterStepSize = stepSize / math.sqrt(historyNum + index + 1)
          //val thisIterStepSize = stepSize / (iter*iter)
          //val thisIterStepSize = stepSize / iter
          val thisIterStepSize = stepSize / math.sqrt(iter)

          userFeatureMatrixTrans(::, i) :+= thisIterStepSize * (predictionError * ProductFeatureMatrixTrans(::, j) - regParam * userFeatureMatrixTrans(::, i))
          ProductFeatureMatrixTrans(::, j) :+= thisIterStepSize * (predictionError * userFeatureMatrixTrans(::, i) - regParam * ProductFeatureMatrixTrans(::, j))

          val predictionError2 = rating - userFeatureMatrixTrans(::, i).t * ProductFeatureMatrixTrans(::, j)
          val improve = predictionError - predictionError2
          //println(s"index:$index step:$thisIterStepSize" + '\n' + s"improve:$improve")
        }

        val afterRMSE = MatrixFactorizationWithSGD.dataArrayRmse(userFeatureMatrixTrans, ProductFeatureMatrixTrans, matrixEntrys)
        println("****************************************************************************")
        println(s"=======Google training RMSE after iter $iter computation: $afterRMSE=======")
        println("****************************************************************************")

        accum += matrixEntrys.length
        List((m, (true, userFeatureMatrixTrans.t)), (n, (false, ProductFeatureMatrixTrans.t)))
    }.persist(intermediateRDDStorageLevel)

    val updatedBlockUserFeatures = updatedFeatures.filter(_._2._1).map(x =>(x._1, x._2._2)).persist(intermediateRDDStorageLevel)
    val updatedBlockProductFeatures = updatedFeatures.filter(!_._2._1).map(x =>(x._1, x._2._2)).persist(intermediateRDDStorageLevel)
    println("updatedBlockUserFeatures.count:" + updatedBlockUserFeatures.count())
    println("updatedBlockProductFeatures.count:" + updatedBlockProductFeatures.count())

    val newBlockUserFeatures = blockUserFeatures.subtractByKey(updatedBlockUserFeatures).union(updatedBlockUserFeatures)
      .persist(intermediateRDDStorageLevel)
    val newBlockProductFeatures = blockProductFeatures.subtractByKey(updatedBlockProductFeatures).union(updatedBlockProductFeatures)
      .persist(intermediateRDDStorageLevel)
    println("newBlockUserFeatures.count:" + newBlockUserFeatures.count())
    println("newBlockProductFeatures.count:" + newBlockProductFeatures.count())
    //println(newBlockUserFeatures.toDebugString)

    stratumWithFeatures.unpersist()
    updatedFeatures.unpersist()
    updatedBlockUserFeatures.unpersist()
    updatedBlockProductFeatures.unpersist()

    println(s"====Google accum.value:"+accum.value+"==========================")

    //(blockUserFeatures,blockProductFeatures,historyNum + accum.value)

    (newBlockUserFeatures, newBlockProductFeatures, historyNum + accum.value)
  }

}

object MatrixFactorizationWithSGD {
  def mapPredictedRating(r: Double, implicitPrefs: Boolean) = {
    if (implicitPrefs) math.max(math.min(r, 1.0), 0.0)
    else r
  }

  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], implicitPrefs: Boolean) = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map { x =>
      ((x.user, x.product), mapPredictedRating(x.rating, implicitPrefs))
    }.join(data.map(x => ((x.user, x.product), x.rating))).values

    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }

  def rddRmse(model: MatrixFactorizationModel, data: RDD[MatrixEntry], implicitPrefs: Boolean = false) = {
    computeRmse(model, data.map{case MatrixEntry(i,j,r) => Rating(i.toInt,j.toInt,r)}, implicitPrefs)
  }

  def dataArrayRmse(userFeatureMatrixTrans: BrzMatrix[Double],ProductFeatureMatrixTrans: BrzMatrix[Double], data: Array[((Int, Int), Double)]) = {
    val rmse = data.map{
      case ((i, j), rating) =>
        val predictionError = rating - userFeatureMatrixTrans(::, i).t * ProductFeatureMatrixTrans(::, j)
        predictionError * predictionError
    }.reduce(_ + _)
    Math.sqrt(rmse / data.length)
  }

  def rddToArrayRmse(model: MatrixFactorizationModel, data: RDD[MatrixEntry]) = {
    val userFeatures = model.userFeatures.map {
      case (i, features) => IndexedRow(i.toLong, new DenseVector(features))
    }
    val productFeatures = model.productFeatures.map {
      case (i, features) => IndexedRow(i.toLong, new DenseVector(features))
    }
    val datas = data.map{
      case MatrixEntry(i,j,k) => ((i.toInt,j.toInt),k)
    }.collect
    dataArrayRmse(new IndexedRowMatrix(userFeatures).toBreeze.t, new IndexedRowMatrix(productFeatures).toBreeze.t, datas)
  }

  //Jump the values not found
  def rddToArrayRmse2(model: MatrixFactorizationModel, data: RDD[MatrixEntry]) = {
    val userFeatures = model.userFeatures.collect.toMap
    println("userFeatures")
    println(userFeatures.keys.toArray.sorted)
    println("max" + userFeatures.keys.toArray.sorted.max)
    println("userFeatures.keys.size:" + userFeatures.keys.size)

    val productFeatures = model.productFeatures.collect.toMap
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

