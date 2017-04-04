package main;

import org.apache.spark.ml.feature.CountVectorizerModel;

/**
 * Created by hadoop on 17-4-3.
 */
interface CLDAModel {
  void BuildCorpus(String source, String output);

  void ConvertComment2Vector(String comment, String output, CountVectorizerModel vectorizerModel);

  void Build(String source, String output, int topicNum);

  void Predict(String comment, String modelUrl, CountVectorizerModel counterModel);
}
