package org.apache.spark.mllib.recommendation

import breeze.optimize.proximal.Constraint._
import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BrzMatrix, DenseVector=>BrzVector, Matrix => BM}

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{DenseVector, DenseMatrix, SparseMatrix, Matrix, Matrices}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, IndexedRow, BlockMatrix, CoordinateMatrix, IndexedRowMatrix}

import org.apache.spark.util.random.BernoulliSampler

import breeze.linalg.shuffle
import java.util.Random
import org.apache.spark.util.random.XORShiftRandom

import scala.reflect.ClassTag

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
    val numCol = blockRatings.numColBlocks
    val accum = blockUserFeatures.sparkContext.accumulator(0)
    val stratumWithFeatures = blockRatings.blocks.filter {
      case ((x, y), matrix) =>
        if (math.abs(x - y) % numCol == stratumID) true else false
    }.map {
      case ((x, y), matrix) => (x, ((x, y), matrix))
    }.leftOuterJoin(blockUserFeatures).map {
      case (k, (((x, y), matrix), userFeature)) => (y, (((x, y), matrix), userFeature))
    }.leftOuterJoin(blockProductFeatures).persist(intermediateRDDStorageLevel)

    println(s"********Yeah! iter:$iter stratumID:$stratumID *********************")
    
    println("=====stratumWithFeatures================")
    println("1 yeah 看看stratumWithFeatures的blockUserFeatures和blockProductFeatures")
    println("stratumWithFeatures.count" + stratumWithFeatures.count)
    stratumWithFeatures.foreach(println)

    val updatedFeatures: RDD[(Int, (Boolean, BrzMatrix[Double]))] = stratumWithFeatures.flatMap {
      case (k, ((((m, n), matrix), optionUserFeatureBlock), optionProductFeatureBlock)) =>
        // compute new userFeature and productFeature
        //TODO: 处理并没有FeatureBlock或者FeatureBlock缺行
        //TODO: 成块新增好处多，避免新增id不连续
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
        /*
        println("2 yeah 看看stratumWithFeatures中一个stratum拆包后的userFeatureMatrix和ProductFeatureMatrix")
        println("====userFeatureMatrixTrans=====================")
        println("rows:" + userFeatureMatrixTrans.rows)
        println("cols:" + userFeatureMatrixTrans.cols)
        */
        /*
        println("====userFeatureMatrix=====================")
        println(userFeatureMatrixTrans.t)
*/
        /*
        println("====ProductFeatureMatrixTrans====================")
        println("rows:" + ProductFeatureMatrixTrans.rows)
        println("cols:" + ProductFeatureMatrixTrans.cols)

        println("====ProductFeatureMatrix====================")
        println(ProductFeatureMatrixTrans.t)
        println("====Data matrix=====================")
        println("rows:" + matrix.numRows)
        println("cols:" + matrix.numCols)
        //println(matrix)
        */

        //SGD的循环
        val matrixEntrys = matrix.toBreeze.activeIterator.toArray
        val shuffleIndices = shuffle(BrzVector.range(0, matrixEntrys.length))
        accum += matrixEntrys.length
        // TODO: 加sqrt会使stepsize比较大，有比较大的机会把不好的局部解冲掉
        println("********************************************")
        println("=======SGD UPDATE shuffled ITEMS============")
        println("*******************************************")

        shuffleIndices.toArray.zipWithIndex.foreach { x =>
          val ((i, j), rating) = matrixEntrys(x._1)
          val index = x._2
          val predictionError = rating - userFeatureMatrixTrans(::, i).t * ProductFeatureMatrixTrans(::, j)

          /*
          //TODO: 确认向量行列对不对，length对不对
          println("*************** before **********************" + '\n'
            + s"i: $i, j: $j" + '\n'
            + s"predictionError: $predictionError, rating: $rating" + '\n'
            + userFeatureMatrixTrans(::, i) + '\n'
            + ProductFeatureMatrixTrans(::, j))
            */

          val thisIterStepSize = stepSize / math.sqrt(historyNum + index)
          println(s"thisIterStepSize: $thisIterStepSize")
          userFeatureMatrixTrans(::, i) :+= thisIterStepSize * (predictionError * ProductFeatureMatrixTrans(::, j) - regParam * userFeatureMatrixTrans(::, i))
          ProductFeatureMatrixTrans(::, j) :+= thisIterStepSize * (predictionError * userFeatureMatrixTrans(::, i) - regParam * ProductFeatureMatrixTrans(::, j))
          val predictionError2 = rating - userFeatureMatrixTrans(::, i).t * ProductFeatureMatrixTrans(::, j)

          /*
          println("*************** after **********************" + '\n'
            + s"i: $i, j: $j" + '\n'
            + s"predictionError: $predictionError, predictionError2:$predictionError2, rating: $rating" + '\n'
            + userFeatureMatrixTrans(::, i) + '\n'
            + ProductFeatureMatrixTrans(::, j))
            */
        }
        List((m, (true, userFeatureMatrixTrans.t)), (n, (false, ProductFeatureMatrixTrans.t)))
    }.persist(intermediateRDDStorageLevel)

    stratumWithFeatures.unpersist()

    val updatedBlockUserFeatures = updatedFeatures.filter(_._2._1).map(x =>(x._1, x._2._2)).persist(intermediateRDDStorageLevel)
    val updatedBlockProductFeatures = updatedFeatures.filter(!_._2._1).map(x =>(x._1, x._2._2)).persist(intermediateRDDStorageLevel)

    //TODO: 参数和data两边都可能缺对应block
    println("yahoo old blockUserFeatures:" + '\n' + blockUserFeatures.count)
    blockUserFeatures.foreach {
      case (i, matrix) =>
        println(s"yahoo old blockUserFeatures i:$i "+'\n'+ matrix.rows+'\t'+ matrix.cols)
        println(matrix)
    }
    println("yahoo updatedBlockUserFeatures:" + '\n' + updatedBlockUserFeatures.count)
    updatedBlockUserFeatures.foreach {
      case (i, matrix) =>
        println(s"yahoo updatedBlockUserFeatures i:$i "+'\n'+ matrix.rows+'\t'+ matrix.cols)
        println(matrix)
    }

    val newBlockUserFeatures = blockUserFeatures.subtractByKey(updatedBlockUserFeatures).union(updatedBlockUserFeatures)
    val newBlockProductFeatures = blockProductFeatures.subtractByKey(updatedBlockProductFeatures).union(updatedBlockProductFeatures)

    println("yahoo new blockUserFeatures:" + '\n' + blockUserFeatures.count)
    newBlockUserFeatures.foreach {
      case (i, matrix) =>
        println(s"yahoo new blockUserFeatures i:$i "+'\n'+ matrix.rows+'\t'+ matrix.cols)
        println(matrix)
    }

    updatedFeatures.unpersist()
    updatedBlockUserFeatures.unpersist()
    updatedBlockProductFeatures.unpersist()
    (newBlockUserFeatures, newBlockProductFeatures, historyNum + accum.value)
  }

  /**
   * Implementation of the ALS algorithm.
   */
  def train(
      ratings: RDD[MatrixEntry],
      rowsPerBlock: Int = 1024,
      colsPerBlock: Int = 1024,
      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): MatrixFactorizationModel = {
    val rank = model.rank
    val userFeatures = model.userFeatures
    val productFeatures = model.productFeatures
    val histroyNum = model.numEntries
    val currentNum = ratings.count
    var sofarNum = histroyNum
    val histroyRate = histroyNum.toDouble / (histroyNum + currentNum)
    val currentRate = currentNum.toDouble / (histroyNum + currentNum)
    println(s"histroyNum: $histroyNum  currentNum: $currentNum")
    println(s"histroyRate: $histroyRate  currentRate: $currentRate")

    //TODO: 看看这个rank作为参数对不对
    //TODO: 根据blockRatings.numRowBlocks和blockRatings.numRows调整blockRatings.rowsPerBlock的大小
    val blockRatings:BlockMatrix = new CoordinateMatrix(ratings).toBlockMatrix(rowsPerBlock, colsPerBlock).persist(intermediateRDDStorageLevel)
    var blockUserFeatures: RDD[(Int, BrzMatrix[Double])] = new IndexedRowMatrix(userFeatures.map(x => IndexedRow(x._1.toLong, new DenseVector(x._2))))
      .toBlockMatrix(rowsPerBlock, rank).blocks.map {
         case ((x, y), featureMatrix) => (x, featureMatrix.toBreeze.toDenseMatrix)
      }.persist(intermediateRDDStorageLevel)
    var blockProductFeatures: RDD[(Int, BrzMatrix[Double])] = new IndexedRowMatrix(productFeatures.map(x => IndexedRow(x._1.toLong, new DenseVector(x._2))))
      .toBlockMatrix(colsPerBlock, rank).blocks.map {
         case ((x, y), featureMatrix) => (x, featureMatrix.toBreeze.toDenseMatrix)
      }.persist(intermediateRDDStorageLevel)

    val numCol = blockRatings.numColBlocks
    val initBlockUserFeatures = blockUserFeatures
    val initBlockProductFeatures = blockProductFeatures

    println("==BlockMatrix==================")
    println("ratings.count:"+ ratings.count)
    //ratings.take(20).foreach(println)
    println("blockRatings.blocks.count:" + blockRatings.blocks.count)
    //TODO: 数据量小的时候这两个量都是1，所以并没有的,还要测试2以及更多
    println("===blockRatings:Row and Col num of blocks=======")
    println(blockRatings.numRowBlocks)
    println(blockRatings.numColBlocks)
    println("===blockRatings:Row and Col num of entrys=======")
    println(blockRatings.numRows)
    println(blockRatings.numCols)
    blockRatings.blocks.foreach(println)
    println("0 yeah 看看blockUserFeatures和blockProductFeatures初始值")
    blockUserFeatures.foreach(println)
    blockProductFeatures.foreach(println)

    //TODO(wli12): 不需要miniBatchFraction 这里就是每次一个元素的sgd，更快，虽然收敛性有待探讨
    //TODO:通过join处理没出现过的new user，这个点还需要测试
    for (iter <- 1 to numIterations) {
      for (stratumID <- 0 until blockRatings.numColBlocks) {
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


    println("=========Yeah Preserve the init model blockUserFeatures============")
    blockUserFeatures = blockUserFeatures.join(initBlockUserFeatures).map {
      case (i, (features, oldFeatures)) =>
        println(s"i: $i")
        println("features")
        println(features)
        println("oldFeatures")
        println(oldFeatures)
        println("updatedUserFeatures")
        println(histroyRate * features + currentRate * oldFeatures)
        (i, histroyRate * features + currentRate * oldFeatures)
    }
    //println("=========Preserve the init model blockProductFeatures============")
    blockProductFeatures = blockProductFeatures.join(initBlockProductFeatures).map{
      case (i, (features, oldFeatures)) =>
        /*
        println(s"i: $i")
        println("features")
        println(features)
        println("oldFeatures")
        println(oldFeatures)
        */
        (i, histroyRate * features + currentRate * oldFeatures)
    }
    println("=========Yeah After join with init blocks================" )


    println("=====updatedUserFeatures=============")
    val updatedUserFeatures = new BlockMatrix(blockUserFeatures.map(x => ((x._1, 0), Matrices.fromBreeze(x._2))), rowsPerBlock, rank)
      .toIndexedRowMatrix().rows.map{
      case IndexedRow(i, features) =>
        (i.toInt, features.toArray)
    }
    println("=====updatedProductFeatures=============")
    val updatedProductFeatures = new BlockMatrix(blockProductFeatures.map(x => ((x._1, 0), Matrices.fromBreeze(x._2))), colsPerBlock, rank)
      .toIndexedRowMatrix().rows.map{
      case IndexedRow(i, features) =>
        (i.toInt, features.toArray)
    }
    println("=====================================")
    println("countByKey:updatedUserFeatures and updatedProductFeatures")
    println(updatedUserFeatures.countByKey().toList.sorted)
    println(updatedProductFeatures.countByKey().toList.sorted)
    println(s"histroyNum: $histroyNum, currentNum:$currentNum,  sofarNum: $sofarNum, currentNumBySofar:" + (sofarNum - histroyNum)/numIterations)
    new MatrixFactorizationModel(rank, updatedUserFeatures, updatedProductFeatures, histroyNum + currentNum)
  }
}
