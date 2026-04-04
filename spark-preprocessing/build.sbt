// spark-preprocessing/build.sbt

name         := "preprocessing"
version      := "1.0.0"
scalaVersion := "2.12.18"

// Classe principale
Compile / mainClass := Some("com.ml.microservices.Pretraitement")
assembly / mainClass := Some("com.ml.microservices.Pretraitement")

// Plugin assembly pour créer un fat JAR
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf"              => MergeStrategy.concat
  case x                             => MergeStrategy.first
}

libraryDependencies ++= Seq(
  // Spark — "provided" car fourni par spark-submit à l'exécution
  "org.apache.spark" %% "spark-core"  % "3.5.1" % "provided",
  "org.apache.spark" %% "spark-sql"   % "3.5.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.5.1" % "provided",

  // Logging (Log4j2 pour Configurator)
  "org.apache.logging.log4j" % "log4j-api"        % "2.20.0",
  "org.apache.logging.log4j" % "log4j-core"       % "2.20.0",
  "org.apache.logging.log4j" % "log4j-slf4j-impl" % "2.20.0",

  // Tests
  "org.scalatest" %% "scalatest" % "3.2.17" % Test
)

// Java AWT headless (pour ImageIO dans le container sans display)
javaOptions += "-Djava.awt.headless=true"
fork := true
