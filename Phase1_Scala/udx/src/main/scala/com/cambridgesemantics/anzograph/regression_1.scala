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

object regression_1 
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
    
    println(xtest.summarize())
    println(ytest.summarize())
    println(xtrain.summarize())
    println(ytrain.summarize())
    
    val trainData_ds = datasetFromTensors(xtrain, "train_dataset")
    val testData_ds = datasetFromTensors(xtest, "test_dataset")
    val trainTarget_ds = datasetFromTensors(ytrain, "train_target")
    val testTarget_ds = datasetFromTensors(ytest, "test_target")
    
    val trainData = trainData_ds.zip(trainTarget_ds)
    val evalTestData = testData_ds.zip(testTarget_ds)
    
    val inp = tf.learn.Input(FLOAT32, Shape(-1, 11))
    val trainInput = tf.learn.Input(FLOAT32, Shape(-1))
    
    logger.info("Building the logistic regression model.")
    
    val layer = tf.learn.Flatten[Float]("Input/Flatten") >>
        tf.learn.Linear[Float]("Layer_0/Linear", 64) >> tf.learn.ReLU[Float]("Layer_0/ReLU", 0.1f) >>
        tf.learn.Linear[Float]("Layer_1/Linear", 64) >> tf.learn.ReLU[Float]("Layer_1/ReLU", 0.1f) >>
        tf.learn.Linear[Float]("OutputLayer/Linear", 1)
        
    val loss = tf.learn.L2Loss[Float, Float]("Loss/L2")

    val optimizer = tf.train.GradientDescent(0.01f)
    
    val summariesDir = Paths.get("tmp/log")
    
    val model = tf.learn.Model.simpleSupervised(
      input = inp,
      trainInput = trainInput,
      layer = layer,
      loss = loss,
      optimizer = optimizer)
      
    logger.info("Training the linear regression model.")
    
    val estimator = InMemoryEstimator(
        model, 
        tf.learn.Configuration(Some(summariesDir)),
        trainHooks = Set(
          tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
          tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(1000))),
          tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir))
    
    estimator.train(() => trainData, StopCriteria(maxSteps = Some(10000)))
    
    logger.info("Training the linear regression model - COMPLETED")
    
    val accMetric = tf.metrics.MapMetric(
      (v: (Output[Float], (Output[Float], Output[Float]))) => {
        (tf.argmax(v._1, -1, INT64).toFloat, v._2._2.toFloat)
      }, tf.metrics.Accuracy("Accuracy"))
    
    val evalOp = estimator.evaluate(
                  () => evalTestData,
                  metrics = Seq(accMetric),
                  maxSteps = -1)
                  
    println(evalOp.summarize())
    
    logger.info("Evaluating linear regression model.")
  } 
  
}