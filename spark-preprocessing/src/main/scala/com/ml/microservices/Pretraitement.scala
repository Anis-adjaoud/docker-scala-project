package com.ml.microservices

import org.apache.spark.sql.{SparkSession, SaveMode}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer
import org.slf4j.LoggerFactory
import java.util.Properties
import java.io.FileInputStream
import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.config.Configurator

object Pretraitement {
  private val logger = LoggerFactory.getLogger(this.getClass.getName)

  def main(args: Array[String]): Unit = {
    Configurator.setLevel("org.apache.spark",  Level.ERROR)
    Configurator.setLevel("org.apache.hadoop", Level.ERROR)
    Configurator.setLevel("com.ml.microservices", Level.INFO)

    logger.info("Démarrage du Microservice de Prétraitement...")

    // -------------------------------------------------------
    // Config : lire config.properties OU variables d'env Docker
    // Les variables d'env ont priorité sur le fichier properties
    // -------------------------------------------------------
    val trainInput  = sys.env.getOrElse("PATH_TRAIN_IN",  readProp("path.train.in"))
    val testInput   = sys.env.getOrElse("PATH_TEST_IN",   readProp("path.test.in"))
    val trainOutput = sys.env.getOrElse("PATH_TRAIN_OUT", readProp("path.train.out"))
    val testOutput  = sys.env.getOrElse("PATH_TEST_OUT",  readProp("path.test.out"))

    logger.info(s"Train input  : $trainInput")
    logger.info(s"Test  input  : $testInput")
    logger.info(s"Train output : $trainOutput")
    logger.info(s"Test  output : $testOutput")

    implicit val spark: SparkSession = SparkSession.builder()
      .appName("IntelImagePreprocess")
      .master("local[*]")
      .getOrCreate()

    Configurator.setLevel("org.apache.spark",  Level.ERROR)
    Configurator.setLevel("org.apache.hadoop", Level.ERROR)

    import spark.implicits._

    logger.info(">>> Phase 1 : Stats AVANT Prétraitement")
    val rawTrain = spark.read.format("binaryFile")
      .option("recursiveFileLookup", "true").load(trainInput)
    val rawTest  = spark.read.format("binaryFile")
      .option("recursiveFileLookup", "true").load(testInput)

    val trainForStats = rawTrain
      .withColumn("path_norm", regexp_replace($"path", "\\\\", "/"))
      .withColumn("label_txt", element_at(split($"path_norm", "/"), -2))

    ImageUtils.showStats(trainForStats, "Entraînement Brut")

    logger.info(">>> Phase 2 : Prétraitement (UDF resize 150x150)")
    val trainProcessed = ImageUtils.baseProcess(rawTrain)
    val testProcessed  = ImageUtils.baseProcess(rawTest)

    ImageUtils.showStats(trainProcessed, "Entraînement après redimensionnement")

    logger.info(">>> Phase 3 : Encodage des labels")
    val indexer = new StringIndexer()
      .setInputCol("label_txt")
      .setOutputCol("label")
      .fit(trainProcessed)

    val trainFinal = indexer.transform(trainProcessed).select("features", "label")
    val testFinal  = indexer.transform(testProcessed).select("features", "label")

    logger.info(">>> Phase 4 : Écriture Parquet")
    trainFinal.coalesce(1).write.mode(SaveMode.Overwrite).parquet(trainOutput)
    testFinal.coalesce(1).write.mode(SaveMode.Overwrite).parquet(testOutput)

    indexer.labelsArray.flatten.zipWithIndex.foreach { case (name, idx) =>
      logger.info(s"Mapping label : $idx -> $name")
    }

    logger.info("Prétraitement terminé avec succès.")
    logger.info("Attente de 60s avant extinction...")
    Thread.sleep(60000)
    spark.stop()
  }

  // Lecture config.properties (fallback si pas de variable d'env)
  private def readProp(key: String): String = {
    val props = new Properties()
    try {
      props.load(new FileInputStream("config.properties"))
      val v = props.getProperty(key)
      if (v == null) throw new RuntimeException(s"Clé '$key' introuvable dans config.properties")
      v
    } catch {
      case e: Exception =>
        logger.error(s"Impossible de lire config.properties : ${e.getMessage}")
        sys.exit(1)
    }
  }
}
