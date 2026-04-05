# =============================================================
#  spark-preprocessing/Dockerfile
#  Image multi-stage : compile le JAR Scala puis l'exécute
# =============================================================

# ---- Stage 1 : compilation ----
FROM sbtscala/scala-sbt:eclipse-temurin-17.0.15_6_1.12.8_3.8.3 AS builder

WORKDIR /build

COPY build.sbt .
COPY project/ project/
RUN sbt update

COPY src/ src/
RUN sbt assembly

# ---- Stage 2 : exécution ----
FROM eclipse-temurin:17-jre-jammy

ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3

RUN apt-get update && apt-get install -y curl && \
    curl -sL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    | tar -xz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${PATH}"

WORKDIR /app

COPY --from=builder /build/target/scala-*/preprocessing-assembly-*.jar app.jar
COPY config.properties .

ENV JAVA_TOOL_OPTIONS="-Djava.awt.headless=true"

ENV PATH_TRAIN_IN=/data/raw/train
ENV PATH_TEST_IN=/data/raw/test
ENV PATH_TRAIN_OUT=/data/processed/train
ENV PATH_TEST_OUT=/data/processed/test

CMD ["spark-submit", \
     "--class", "com.ml.microservices.Pretraitement", \
     "--master", "local[*]", \
     "--driver-memory", "4g", \
     "--conf", "spark.driver.extraJavaOptions=-Djava.awt.headless=true", \
     "app.jar"]
