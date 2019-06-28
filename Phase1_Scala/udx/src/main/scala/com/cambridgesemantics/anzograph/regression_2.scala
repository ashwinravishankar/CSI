package com.cambridgesemantics.anzograph

import org.platanios.tensorflow.api._
import org.platanios.tensorflow._
import scala.io.Source._
import org.platanios.tensorflow.api.ops.data.Data._
import org.platanios.tensorflow.api.implicits
import org.platanios.tensorflow.api.implicits._
import org.platanios.tensorflow.api.implicits.helpers
import org.platanios.tensorflow.api.implicits.helpers._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToDataType, OutputToShape}
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.core.types.{INT64, STRING, Variant}
import org.platanios.tensorflow.api.implicits.helpers._
import org.platanios.tensorflow.api.io.{CompressionType, NoCompression}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.learn.estimators._
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.ops.metrics
import org.platanios.tensorflow.api.ops.Output

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import org.platanios.tensorflow.api.learn.layers.Input

import java.nio.file.{Files, Paths}
import java.nio.file.FileSystem

object regression_2 
{
    
  // Implicit helpers for Scala 2.11.
  implicit val evOutputStructureFloatLong : OutputStructure[(Output[Float], Output[Long])]  = examples.evOutputStructureFloatLong
  implicit val evOutputToDataTypeFloatLong: OutputToDataType[(Output[Float], Output[Long])] = examples.evOutputToDataTypeFloatLong
  implicit val evOutputToShapeFloatLong   : OutputToShape[(Output[Float], Output[Long])]    = examples.evOutputToShapeFloatLong

  def main(args: Array[String]) 
  {
    val logger = Logger(LoggerFactory.getLogger("Linear Regression"))
    
    // Using python to export Train / Test datasets (including target value) as .NPY files. 
    // As the input dataset is from SPARQL query, it does not matter how the tensors are read.
    // Python code that created the .NPY files  ===>  /Users/ashwinravishankar/Work/tmp_foo/udx/csv_to_npy.py
    
    
    val xtrain = (Tensor.fromNPY[Double](Paths.get("/Users/ashwinravishankar/Work/WineQuality/Dataset/NPY/train.npy"))).toFloat
    val ytrain = (Tensor.fromNPY[Long](Paths.get("/Users/ashwinravishankar/Work/WineQuality/Dataset/NPY/train_target.npy"))).toFloat
    val xtest = (Tensor.fromNPY[Double](Paths.get("/Users/ashwinravishankar/Work/WineQuality/Dataset/NPY/test.npy"))).toFloat
    val ytest = (Tensor.fromNPY[Long](Paths.get("/Users/ashwinravishankar/Work/WineQuality/Dataset/NPY/test_target.npy"))).toFloat
    val t_stats = (Tensor.fromNPY[Double](Paths.get("/Users/ashwinravishankar/Work/WineQuality/Dataset/NPY/t_stats.npy"))).toFloat
  
//    println(ytest.summarize())
//    println(ytest(0))
    
    val inputs = tf.placeholder[Float](Shape(1, 11))
    val outputs = tf.placeholder[Float](Shape(1, 1))
    val weights = tf.variable[Float]("weights", Shape(11, 1), tf.ZerosInitializer)
    val predictions = tf.matmul(inputs, weights)
    val loss = tf.sum(tf.square(tf.subtract(predictions, outputs)))
    val trainOp = tf.train.GradientDescent(0.01f).minimize(loss)  //tf.train.AdaGrad(1.0f).minimize(loss)  
    
    logger.info("Training the linear regression model.")
    val sess = Session()
    sess.run(targets = tf.globalVariablesInitializer())
    for (i <- 0 to 50) {
      var trainLoss = Tensor[Float](Shape(-1))  //tf.placeholder[Float](Shape(-1))
      for (j <- 0 to 1278) {
        val feeds = Map(inputs -> (xtrain(j)).reshape(Shape(1,11)), outputs -> ytrain(j))
        trainLoss = sess.run(feeds = feeds, fetches = loss, targets = trainOp)
      }
      logger.info(s"Train loss at iteration $i = ${trainLoss.scalar} ")
    }
    logger.info(s"Trained weight value: ${sess.run(fetches = weights.value).summarize()}")
    logger.info(s"True weight value: $weights\n\n\n")
    
//    weights.toTensor.writeNPY(Paths.get("/Users/ashwinravishankar/Work/WineQuality/Dataset/NPY/weights.npy"))
    logger.info(s"Prediction Phase ---- True Output ----- Predicted Output")
    for (i <- 0 to 318) {
      val feeds = Map(inputs -> (xtest(i)).reshape(Shape(1, 11)))
      val prediction = sess.run(feeds = feeds, fetches = predictions)
      val trueop = ytest(i)
      logger.info(s"$trueop  \t ${prediction.scalar}")
    }
    
    
  } 
  
}