/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.examples.mllib

import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser
import breeze.optimize.proximal.Constraint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
 * An example app for ALS on MovieLens data (http://grouplens.org/datasets/movielens/).
 * Run with
 * {{{
 * bin/run-example org.apache.spark.examples.mllib.MovieLensALS
 * }}}
 * A synthetic dataset in MovieLens format can be found at `data/mllib/sample_movielens_data.txt`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object MovieLensALS {
  case class Params(
    input: String = null,
    productNamesFile: String = null,
    kryo: Boolean = false,
    numIterations: Int = 20,
    userConstraint: String = "SMOOTH",
    productConstraint: String = "SMOOTH",
    userLambda: Double = 1.0,
    productLambda: Double = 1.0,
    rank: Int = 10,
    delimiter: String = "::",
    numUserBlocks: Int = -1,
    numProductBlocks: Int = -1,
    implicitPrefs: Boolean = false,
    autoParams: Boolean = false,
    testStreaming: Boolean = false,
    sgdStepSize: Double = 0.01,
    sgdNumIterations: Int =  5,
    sgdRegParam: Double = 1.0,
    rowsPerBlock: Int = 1024,
    colsPerBlock: Int = 1024,
    sgdDataRate: Double = 0.75,
    timeStamp: Boolean = false) extends AbstractParams[Params]

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
      arg[String]("<input>")
        .required()
        .text("input paths to a MovieLens dataset of ratings")
        .action((x, c) => c.copy(input = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class org.apache.spark.examples.mllib.MovieLensALS \
          |  examples/target/scala-*/spark-examples-*.jar \
          |  --rank 5 --numIterations 20 --userConstraint SMOOTH --productConstraint SMOOTH --userLambda 0.01 --productLambda 1.0 --kryo\
          |  data/mllib/sample_movielens_data.txt
          |
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"MovieLensALS with $params")
    if (params.kryo) {
      conf.registerKryoClasses(Array(classOf[mutable.BitSet], classOf[Rating]))
        .set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val implicitPrefs = params.implicitPrefs
    val delimiter = params.delimiter
    
    val ratings = sc.textFile(params.input).map { line =>
      val fields = line.split(delimiter)
      if (implicitPrefs) {
        /*
         * MovieLens ratings are on a scale of 1-5:
         * 5: Must see
         * 4: Will enjoy
         * 3: It's okay
         * 2: Fairly bad
         * 1: Awful
         * So we should not recommend a movie if the predicted rating is less than 3.
         * To map ratings to confidence scores, we use
         * 5 -> 2.5, 4 -> 1.5, 3 -> 0.5, 2 -> -0.5, 1 -> -1.5. This mappings means unobserved
         * entries are generally between It's okay and Fairly bad.
         * The semantics of 0 in this expanded world of non-positive weights
         * are "the same as never having interacted at all".
         */
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble - 2.5)
      } else {
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
      }
    }.persist(StorageLevel.MEMORY_AND_DISK)

    val numRatings = ratings.count()
    val numUsers = ratings.map(_.user).distinct().count()
    val numMovies = ratings.map(_.product).distinct().count()

    println(s"Got $numRatings ratings from $numUsers users on $numMovies movies.")
    
    val splits = ratings.randomSplit(Array(0.8, 0.2), 1L)
    val training = splits(0).persist(StorageLevel.MEMORY_AND_DISK)
    
    val test = if (params.implicitPrefs) {
      /*
       * 0 means "don't know" and positive values mean "confident that the prediction should be 1".
       * Negative values means "confident that the prediction should be 0".
       * We have in this case used some kind of weighted RMSE. The weight is the absolute value of
       * the confidence. The error is the difference between prediction and either 1 or 0,
       * depending on whether r is positive or negative.
       */
      splits(1).map(x => Rating(x.user, x.product, if (x.rating > 0) 1.0 else 0.0))
    } else {
      splits(1)
    }.persist(StorageLevel.MEMORY_AND_DISK)


    val numTraining = training.count()
    val numTest = test.count()
    println(s"Training: $numTraining, test: $numTest.")

    ratings.unpersist()

    val bestParams = if (!params.autoParams) params else getBestParamsWithCV(training, params)

    val userConstraint = Constraint.withName(bestParams.userConstraint)
    val productConstraint = Constraint.withName(bestParams.productConstraint)

    val als = new ALS()
      .setRank(bestParams.rank)
      .setIterations(bestParams.numIterations)
      .setUserConstraint(userConstraint)
      .setProductConstraint(productConstraint)
      .setUserLambda(bestParams.userLambda)
      .setProductLambda(bestParams.productLambda)
      .setImplicitPrefs(bestParams.implicitPrefs)
      .setUserBlocks(bestParams.numUserBlocks)
      .setProductBlocks(bestParams.numProductBlocks)

    println(s"Quadratic minimization userConstraint ${userConstraint} productConstraint ${productConstraint}")

    val model = als.run(training)

    //StreamingMovieLensSGD.printModel(model, "ALS Full data model")

    val rmse = computeRmse(model, test, params.implicitPrefs)

    println(s"Test RMSE = $rmse.")

    // create a naive baseline and compare it with the best model
    val meanRating = training.map(_.rating).mean
    val baselineRmse =
      math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating)).mean)
    val improvement = (baselineRmse - rmse) / baselineRmse * 100
    println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    if (params.productNamesFile != null) {
      topicsAnalysis(model, params.productNamesFile)
    }
    zeroRateAnalysis(model)

    val output = s"${params.input}_model/rank${params.rank}sparse${params.productLambda}iter${params.numIterations}.model"
    println(s"Model saving to ${output}")
    model.save(sc, output)

    sc.stop()
  }

  def getBestParamsWithCV(trainingAndValidation: RDD[Rating], params: Params): Params = {
    val splits = trainingAndValidation.randomSplit(Array(0.8, 0.2), 1L)
    val training = splits(0).persist(StorageLevel.MEMORY_AND_DISK)
    val validation = splits(1).persist(StorageLevel.MEMORY_AND_DISK)

    /*
    val numTraining = training.count()
    val numValidation = validation.count()
    println(s"Training: $numTraining, validation: $numValidation")
    */
    val userConstraint = Constraint.withName(params.userConstraint)
    val productConstraint = Constraint.withName(params.productConstraint)

    val numIters = List(10, 20)
    val userLambdas = List(0.0001, 0.01, 0.1, 1.0)
    val productLambdas = List(0.0001, 0.01, 0.1, 1.0)
    val ranks = List(8, 16)

    var bestValidationRmse = Double.MaxValue
    var bestNumIter = -1
    var bestUserLambda = -1.0
    var bestProductLambda = -1.0
    var bestRank = 0

    for (numIter <- numIters; userLambda <- userLambdas; productLambda <- productLambdas; rank <- ranks) {
      val als = new ALS()
        .setRank(rank)
        .setIterations(numIter)
        .setUserConstraint(userConstraint)
        .setProductConstraint(productConstraint)
        .setUserLambda(userLambda)
        .setProductLambda(productLambda)
        .setImplicitPrefs(params.implicitPrefs)
        .setUserBlocks(params.numUserBlocks)
        .setProductBlocks(params.numProductBlocks)

      println(s"Quadratic minimization userConstraint ${userConstraint} productConstraint ${productConstraint}")

      val model = als.run(training)

      val validationRmse = computeRmse(model, validation, false)

      println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
        + rank + ", userLambda = " + userLambda + ", productLambda = " + productLambda + ", and numIter = " + numIter + ".")
      if (validationRmse < bestValidationRmse) {
        bestValidationRmse = validationRmse
        bestNumIter = numIter
        bestUserLambda = userLambda
        bestProductLambda = productLambda
        bestRank = rank
      }
    }

    println("The best model was trained with rank = " + bestRank + ", userLambda = " + bestUserLambda + ", productLambda = " + bestProductLambda
      + ", and numIter = " + bestNumIter + ", and its RMSE on the validation set is " + bestValidationRmse + ".")

    Params(
      params.input,
      params.productNamesFile,
      params.kryo,
      bestNumIter,
      params.userConstraint,
      params.productConstraint,
      bestUserLambda,
      bestProductLambda,
      bestRank,
      params.delimiter,
      params.numUserBlocks,
      params.numProductBlocks,
      params.implicitPrefs,
      params.autoParams)
  }

  def zeroRateAnalysis(model: MatrixFactorizationModel): Unit = {
    val userFeatures = model.userFeatures
    val productFeatures = model.productFeatures
    val sc = productFeatures.sparkContext

    println("===userFeatures====================================")
    val zeroNum = sc.accumulator(0)
    val totalNum = sc.accumulator(0)
    userFeatures.foreach{ x =>
      //println(x._1)
      x._2.foreach{
        x2 =>
          // print(x2.toString + " ")
          if(math.abs(x2) < 1e-7) zeroNum += 1
          totalNum += 1
      }
    }
    val userzerorate = zeroNum.value.toDouble/totalNum.value

    println("===productFeatures====================================")
    zeroNum.setValue(0)
    totalNum.setValue(0)
    productFeatures.foreach{ x =>
      //println(x._1)
      x._2.foreach{
        x2 =>
          //print(x2.toString + " ")
          if(math.abs(x2) < 1e-7) zeroNum += 1
          totalNum += 1
      }
    }
    val itemzerorate = zeroNum.value.toDouble/totalNum.value
    println
    println("userFeatures zero rate:" + userzerorate)
    println("productFeatures zero rate:" + itemzerorate)
  }

  // 有collect，一个product大小的map，一个productFeatures大小（12gpubmed中只有116.8mb所以不怕）
  def topicsAnalysis(model: MatrixFactorizationModel, productNamesFile: String): Unit = {
    val rank = model.rank
    val productFeatures = model.productFeatures
    val sc = productFeatures.sparkContext
    val productNames = sc.textFile(productNamesFile).map{
      line =>
        val Array(productID, productName) = line.split("\t")
        (productID.toInt, productName)
    }.collect().toMap

    val topics = new Array[ArrayBuffer[(Int, Double)]](rank).map(x => new ArrayBuffer[(Int, Double)])
    productFeatures.collect().map {
      case (productId, weights) =>
        for (i <- 0 until weights.length) {
          topics(i).append((productId, weights(i)))
        }
    }

    var count = 0
    topics.foreach {
      topic :ArrayBuffer[(Int, Double)] =>
        count += 1
        println("topic " + count)
        topic.sortWith((x, y) => math.abs(x._2) > math.abs(y._2)).map {
          case (productID, weight) =>
            print(productNames(productID) + ":" + weight + "\t")
        }
        println
    }
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], implicitPrefs: Boolean) = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map { x =>
      ((x.user, x.product), mapPredictedRating(x.rating, implicitPrefs))
    }.join(data.map(x => ((x.user, x.product), x.rating))).values

    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }

  def mapPredictedRating(r: Double, implicitPrefs: Boolean) = {
    if (implicitPrefs) math.max(math.min(r, 1.0), 0.0)
    else r
  }
}
