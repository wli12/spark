package org.apache.spark.mllib.recommendation

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.streaming.dstream.DStream


//抛开原有RDD partition的设计
@Experimental
class StreamingMatrixFactorizationWithSGD private[mllib]  (
  private var numUserBlocks: Int,
  private var numProductBlocks: Int,
  private var stepSize: Double,
  private var numIterations: Int,
  private var regParam: Double)
extends Logging with Serializable {

  def this() = this(-1, -1, 0.1, 10, 0.1)

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

  /** The model to be updated and used for prediction. */
  protected var model: MatrixFactorizationModel = null

  /** The algorithm to use for updating. */
  protected val algorithm: MatrixFactorizationWithSGD = new MatrixFactorizationWithSGD(numUserBlocks, numProductBlocks, stepSize, numIterations, regParam)

  /** The number of elements seen by the current model.*/
  protected var numEntries: Long = 0

  /** Return the latest model. */
  def latestModel(): MatrixFactorizationModel = {
    assertInitialized()
    model
  }

  /** Set the initial weights. Default: [0.0, 0.0]. */
  def loadInitialWeights(matrixFactorizationModel: MatrixFactorizationModel): this.type = {
    this.model = matrixFactorizationModel
    this
  }

  /** Check whether cluster centers have been initialized. */
  private[this] def assertInitialized(): Unit = {
    if (model == null) {
      throw new IllegalStateException(
        "Initial Matrix Factorization Model must be set before starting trainings or predictions")
    }
  }

  /**
  * Update the model by training on batches of data from a DStream.
    * This operation registers a DStream for training the model,
  * and updates the model based on every subsequent
  * batch of data from the stream.
    *
  * @param data DStream containing labeled data
    */
  def run(data: DStream[MatrixEntry],
    rowsPerBlock: Int = 1024,
    colsPerBlock: Int = 1024): Unit = {
    assertInitialized()
    data.foreachRDD { (rdd, time) =>
      if (rdd.count != 0)
        // TODO : 这里使用的是merged model
        model = algorithm.loadInitialWeights(model).train(rdd, rowsPerBlock, colsPerBlock)._2
    }
  }
}
