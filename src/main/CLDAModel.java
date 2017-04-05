package main;

import org.apache.spark.ml.feature.CountVectorizerModel;

/**
 * Created by hadoop on 17-4-3.
 */
interface CLDAModel {
  void BuildCorpus(String input, String output);

  void ConvertLine2Vector(String line, String output, CountVectorizerModel vectorizerModel);

  void Build(String input, String output, int topicNum);

  void Predict(String line, String modelURI, CountVectorizerModel counterModel, int number);
}
