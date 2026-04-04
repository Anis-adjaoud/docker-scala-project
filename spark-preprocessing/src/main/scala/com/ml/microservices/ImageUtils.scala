package com.ml.microservices

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.slf4j.LoggerFactory
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.ByteArrayInputStream

object ImageUtils {
  private val logger = LoggerFactory.getLogger(this.getClass.getName)

  private val preprocessImage = udf((content: Array[Byte]) => {
    try {
      val bais = new ByteArrayInputStream(content)
      val originalImage = ImageIO.read(bais)
      val IMG_SIZE = 150

      val resizedImage = new BufferedImage(IMG_SIZE, IMG_SIZE, BufferedImage.TYPE_INT_RGB)
      val g = resizedImage.createGraphics()
      g.drawImage(originalImage, 0, 0, IMG_SIZE, IMG_SIZE, null)
      g.dispose()

      val pixels = new Array[Double](IMG_SIZE * IMG_SIZE * 3)
      var index = 0
      for (y <- 0 until IMG_SIZE) {
        for (x <- 0 until IMG_SIZE) {
          val color = resizedImage.getRGB(x, y)
          pixels(index)     = ((color >> 16) & 0xFF) / 255.0
          pixels(index + 1) = ((color >> 8)  & 0xFF) / 255.0
          pixels(index + 2) = (color         & 0xFF) / 255.0
          index += 3
        }
      }
      pixels
    } catch {
      case e: Exception =>
        logger.warn(s"=== Error image traitement : $e ===")
        Array.empty[Double]
    }
  })

  def showStats(df: DataFrame, message: String): Unit = {
    logger.info(s"=== Statistiques : $message ===")
    val stats = df.groupBy("label_txt")
      .agg(count("*").as("nb_images"))
      .orderBy(desc("nb_images"))
    stats.show()
    logger.info(s"Total : ${df.count()} images.")
  }

  def baseProcess(df: DataFrame)(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    df.withColumn("path_norm", regexp_replace($"path", "\\\\", "/"))
      .withColumn("label_txt", element_at(split($"path_norm", "/"), -2))
      .withColumn("features", preprocessImage($"content"))
      .filter(size($"features") > 0)
  }
}
