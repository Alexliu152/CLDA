package main;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.logging.Logger;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.ml.clustering.LocalLDAModel;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import org.apdplat.word.WordSegmenter;
import org.apdplat.word.segmentation.Word;
import org.apdplat.word.util.WordConfTools;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import scala.Tuple2;

/**
 * Created by hadoop on 17-4-3.
 */

public class CLDA implements CLDAModel {

  private static Logger logger;
  /* Configuration */
  private Properties config;
  private SparkSession session;
  private String[] vocabulary;


  public CLDA() {

    logger = Logger.getLogger(CLDA.class.getName());

    session = SparkSession.builder().appName("CLDA").master("local").getOrCreate();

    config = new Properties();


    // <editor-fold desc="Read Configuration">
    try {
      config.load(new InputStreamReader(
          CLDA.class.getClassLoader().getResourceAsStream("clda.properties"), "UTF-8"));
    } catch (FileNotFoundException e) {
      logger.warning("Properties File Not found");
    } catch (IOException e) {
      logger.warning("Properties File Read error");
    }
    // </editor-fold>

    // <editor-fold desc="Create files">
    try {
      File comment = new File(config.getProperty("Comment"));
      if (!comment.exists())
        comment.mkdirs();
      comment.createNewFile();
      File segmentation = new File(config.getProperty("Segmentation"));
      if (!segmentation.exists())
        segmentation.mkdirs();
      segmentation.createNewFile();
      File vectors = new File(config.getProperty("Vectors"));
      if (!vectors.exists())
        vectors.mkdirs();
      vectors.createNewFile();
      File vocabulary1 = new File(config.getProperty("Vocabulary"));
      if (!vocabulary1.exists())
        vocabulary1.mkdirs();
      vocabulary1.createNewFile();
    } catch (IOException ioe) {
      logger.warning("Create file failed");
    }
    // </editor-fold>

    // <editor-fold desc="Load Vocabulary">
    FileInputStream fileInputStream = null;
    try {
      fileInputStream = new FileInputStream(config.getProperty("Vocabulary"));
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
    ObjectInputStream in = null;
    try {
      in = new ObjectInputStream(fileInputStream);
    } catch (IOException e) {
      logger.info("Emmpty vocabulary");
    }


    try {
      vocabulary = (String[]) in.readObject();
    } catch (IOException e) {
      // TODO
    } catch (ClassNotFoundException e) {
      // TODO
    } catch (NullPointerException ne) {
      // TODO
    }
    // </editor-fold>

    WordConfTools.set("stopwords.path", config.getProperty("Stopwords"));
  }

  /* Build the corpus for later use rom json files to line-segmented document */


  private static File[] ListDir(String sourceDir) {

    File inputDir = new File(sourceDir);

    if (!inputDir.exists())
      throw new NullPointerException("Input directory does not exist: " + sourceDir);
    if (!inputDir.isDirectory())
      throw new NullPointerException("Input directory not found: " + sourceDir);

    return inputDir.listFiles();
  }

  private static String SegmentComment(String comment) {
    String result = "";
    try {
      List<Word> parsedAnswer = WordSegmenter.seg(comment);

      if (parsedAnswer.size() <= 100)
        return result;

      for (Word word : parsedAnswer) {
        result += " " + word.getText();
      }

    } catch (NullPointerException npe) {
      logger.info("Skip this comment:" + comment);
    }
    return result + "\n";
  }

  public static void main(String[] args) {

    CLDA test = new CLDA();

    tests testCase = tests.SEG;
    String comment =
        "下面分享下我自己的大白话跑步减肥经历吧，之前在hp上发过，希望对想减肥的普通人有些帮助：我曾有过一段成功的减肥经历，但由于回国后猛吃反弹回原始体重。今年4月份重新开始减肥，那时没量体重，不过估计接近巅峰100kg，现在今天早上量是79kg（本人身高180cm，24岁）中间8个月时间，5月份和10月份两个假期由于出去玩停止了加一起有1个多月时间。（今年新年时已经是73kg左右了，达到合理范围，截至回答这个问题，一直维持。）我是在健身房自己健身减肥，计划是每周去4天，一般是周二，三，五，日，但一般较少去四天，多数是3天，也有一部分只去2天。穿的就是普通nike跑鞋，随便买的，可能即使之前大学胖的时候也经常打球，所以一直没感觉到运动减肥造成诸如膝盖疼等哪些不适。健身主要项目就是每天40分钟跑步，一开始是跑步机10速，一口气只能跑3km，然后休息5分钟接着跑。现在一般跑45分钟，11速一口气跑5km，总共跑7-8km的样子，周日时间多一般跑1小时10km左右。然后加30min左右的力量练习，主要是辅助减肥，还没到练肌肉的时候，准备到75kg开始加强局部肌肉练习。然后如果时间来的急，加约30分钟游泳放松，一般就是25m泳道10-20个来回。关于饮食，早上就是牛奶麦片鸡蛋，中午随便吃，晚上在锻炼日少吃一般是只吃水果或者番茄。但是非锻炼日，晚上会随机吃些东西，周末的话也是放开了吃，由于和老婆都是馋猫，所以想吃什么吃什么。对于减肥我觉着，尤其是刚开始，别对自己太狠，什么都不敢吃，不要过于追求速度，那样减下来也会反弹。在周末给自己放纵的机会，想吃什么吃什么。也不要太频繁量体重，早晚体重差别是很大，一般可以到1-2kg，早上是最轻的。拿我举例，有时可能自己一周锻炼了4天，但是体重竟然没掉，有时来了朋友，一周没锻炼，每天都下馆子竟然还没涨，所以减肥是很长期的事情，频繁量体重没什么好处，只要整体趋势是下降就ok了。减肥前期是最容易放弃的，一旦养成习惯，健身就是很理所应当的事了。在减肥过程中，除了需要长期坚持自己的计划外，要给自己打破计划放松的机会。听人家说每周要有天使日，就是那天可以不锻炼，随便吃，这样才能每周都有一个念想。最后提醒大家注意健身中的安全问题吧，也是发生在我本人身上的一个例子。我曾经在有一次跑步过程中，正是加速冲刺的时候接了老婆一个电话，没等气息平稳就说话，有一口气没喘上来，直接头晕过去，差点在健身房摔倒，后来缓了一阵才缓过来，洗澡时还有头晕呕吐感，再后来我一段时间停止了健身，就偶尔会出现心脏早搏的现象（不知道是不是这个名词，没办法描述，就是可以感到自己心脏像打嗝一样的感觉，也求教有没有医学帝可以解答这是什么症状），后来回国后由于在帝都附近空气污染严重，这种现象就更加频繁了，时不时就会出现。直到今年4月恢复国内健身，逐渐这种现象再次消失，到现在已经不再出现了。健身可以带给我们健康，但也要注意健身中的安全。有什么问题可以问我，希望尽我所能给大家提供帮助。";
    CountVectorizerModel counterModel;

    // <editor-fold desc="Test cases">
    switch (testCase) {
      case SEG:
        test.BuildCorpus(test.config.getProperty("Resource"),
            test.config.getProperty("Segmentation"));
        break;
      case VEC:
        test.ConvertCorpus2Vector(test.config.getProperty("Segmentation"),
            test.config.getProperty("Vectors"), test.config.getProperty("CVModel"), 2000);
        break;
      case COM:
        counterModel = CountVectorizerModel.load(test.config.getProperty("CVModel"));
        test.ConvertComment2Vector(comment, test.config.getProperty("Comment"), counterModel);
        break;
      case BID:
        test.Build(test.config.getProperty("Vectors"), test.config.getProperty("LDAModel"), 10);
        break;
      case PRE:
        counterModel = CountVectorizerModel.load(test.config.getProperty("CVModel"));
        test.Predict(comment, test.config.getProperty("LDAModel"), counterModel);
        break;
      default:
        break;
    }
    // </editor-fold>

    System.exit(0);
  }


  /* Transfer each segmented comment to its vector representation */

  @Override
  public void BuildCorpus(String source_dir, String output_file) {

    File[] inputFiles = ListDir(source_dir);
    JsonParser parser = new JsonParser();

    FileOutputStream fileOutputStream = null;
    try {
      fileOutputStream = new FileOutputStream(output_file);
    } catch (FileNotFoundException e) {
      logger.warning("Corpus output file error");
      return;
    }

    // <editor-fold desc="Convert">
    try {
      for (File file : inputFiles) {
        if (file.isFile()) {
          logger.info("Parsing file: " + file.getName());
          JsonArray jsonArray = (JsonArray) parser.parse(new FileReader(file));

          for (JsonElement jsonElement : jsonArray) {
            if (jsonElement.isJsonObject()) {
              JsonObject object = (JsonObject) jsonElement;

              /*
               * Each question form zhihu contains at most 99 answers. Take tha answer into
               * consideration as long as it has more than 50 answers and each answer is longer than
               * 50.
               *
               * Then extract noun and verb from the answer to the result.txt file
               *
               */

              for (int index = 1; index <= 99; index++) {
                JsonElement answer = object.get("answer" + String.valueOf(index));
                if (answer == null)
                  break;

                String answerLine = answer.getAsString();

                if (answerLine != null && answerLine.length() >= 50) {
                  String segmentedComment = SegmentComment(answerLine);

                  if (!segmentedComment.equals("")) {
                    fileOutputStream.write(segmentedComment.getBytes());
                  }
                }
              }

            }
          }
        }
      }
    } catch (FileNotFoundException ne) {
      logger.warning("File not found");
    } catch (IOException e) {
      logger.warning("Writing error");
    }

    // </editor-fold>

    try {
      if (fileOutputStream != null) {
        fileOutputStream.close();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    System.exit(1);
  }

  public void ConvertSegment2Row(String segmentation, ArrayList<Row> data) {
    data.add(RowFactory.create(Arrays.asList(segmentation.split(" "))));
  }

  public void ConvertComment2Vector(String comment, String output,
      CountVectorizerModel vectorizerModel) {

    List<Word> parsedAnswer = WordSegmenter.segWithStopWords(comment);

    StringBuilder stringBuilder = new StringBuilder();

    for (Word word : parsedAnswer) {
      stringBuilder.append(word.getText() + " ");
    }

    String segmentedComment = stringBuilder.toString();

    ArrayList<Row> data = new ArrayList<>();
    ConvertSegment2Row(segmentedComment, data);
    // <editor-fold desc="Schema">
    StructType schema = new StructType(new StructField[] {new StructField("words",
        new ArrayType(DataTypes.StringType, true), false, Metadata.empty())});
    // </editor-fold>
    Dataset<Row> commentData = session.createDataFrame(data, schema);
    Dataset<Row> vectorizedComment = vectorizerModel.transform(commentData);
    Row resultRow = vectorizedComment.select("vector").first();


    Vector features = resultRow.getAs(0);

    System.out.println(features.toSparse().toString());

    String[] featureStringVector = features.toDense().toString().replace(",", " ").replace("[", "")
        .replace("]", "").split(" ");

    // <editor-fold desc="Dense Vector String">
    StringBuilder stringBuilder1 = new StringBuilder();
    stringBuilder1.append(String.valueOf(0)).append(" ");
    int index2 = 1;
    for (String number : featureStringVector) {
      stringBuilder1.append(String.valueOf(index2)).append(":").append(String.valueOf(((Double) Double.parseDouble(number)).intValue())).append(" ");
      index2++;
    }
    // </editor-fold>
    String vector = stringBuilder1.toString();
    FileOutputStream fileOutputStream = null;
    try {
      fileOutputStream = new FileOutputStream(output);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
    try {
        if (fileOutputStream != null) {
            fileOutputStream.write(vector.getBytes());
        }
    } catch (IOException e) {
      e.printStackTrace();
    }
    try {
      fileOutputStream.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void ConvertCorpus2Vector(String sourceDir, String output, String model_url,
      int vocabSize) {

    /*
     *
     * Convert list to dataset in spark
     *
     */

    ArrayList<Row> data = new ArrayList<>();
    File file = new File(sourceDir);

    BufferedReader bufferedReader;
    try {
      bufferedReader = new BufferedReader(new FileReader(file));
      String temp;
      int count = 0; // Count to 5000
      while ((temp = bufferedReader.readLine()) != null) {
        ConvertSegment2Row(temp, data);
        if (++count == 5000)
          break;
      }
    } catch (FileNotFoundException e) {
      logger.warning("Segmentation file not found");
    } catch (IOException e) {
      logger.warning("Read segmentation result error");
    }


    StructType schema = new StructType(new StructField[] {new StructField("words",
        new ArrayType(DataTypes.StringType, true), false, Metadata.empty())});

    Dataset<Row> sentenceData = session.createDataFrame(data, schema);

    /* Vectorize the training data */

    CountVectorizer vectorizer = new CountVectorizer().setInputCol("words").setOutputCol("vector")
        .setVocabSize(vocabSize).setMinTF(2);

    CountVectorizerModel vectorizerModel = vectorizer.fit(sentenceData);

    try {
      vectorizerModel.write().overwrite().save(model_url);
    } catch (IOException e) {
      logger.warning("Counter model save error");
    }


    vocabulary = vectorizerModel.vocabulary();

    FileOutputStream fileOutputStream = null;
    try {
      fileOutputStream = new FileOutputStream(config.getProperty("Vocabulary"));
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
    ObjectOutputStream in = null;
    try {
      in = new ObjectOutputStream(fileOutputStream);
    } catch (IOException e) {
      e.printStackTrace();
    }
    try {
        if (in != null) {
            in.writeObject(vocabulary);
        }
    } catch (IOException e) {
      e.printStackTrace();
    }

    FileOutputStream fileOutputStream2 = null;
    try {
      fileOutputStream2 = new FileOutputStream(output);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }

    Dataset<Row> vectorizedData = vectorizerModel.transform(sentenceData);


    /* Output the data */

    int index = 0;
    for (Row r : vectorizedData.select("vector").collectAsList()) {
      Vector features = r.getAs(0);

      String[] featureStringVector = features.toDense().toString().replace(",", " ")
          .replace("[", "").replace("]", "").split(" ");

      // <editor-fold desc="Dense Vector String">
      StringBuilder stringBuilder1 = new StringBuilder();
      stringBuilder1.append(String.valueOf(0)).append(" ");
      int index2 = 1;
      for (String number : featureStringVector) {
        stringBuilder1.append(String.valueOf(index2)).append(":").append(String.valueOf(((Double) Double.parseDouble(number)).intValue())).append(" ");
        index2++;
      }
      // </editor-fold>

      try {
        fileOutputStream2.write(stringBuilder1.append("\n").toString().getBytes());
      } catch (IOException e) {
        e.printStackTrace();
      }

    }

    try {
      fileOutputStream2.close();
    } catch (IOException e) {
      logger.warning("File close error");
    }
  }

  /* Build the model */
  @Override
  public void Build(String source, String output, int topicNum) {
    // Loads data.
    Dataset<Row> dataSet = session.read().format("libsvm").load(source);
    dataSet.show();

    // Trains a LDA model.
    LDA lda = new LDA().setK(topicNum).setMaxIter(10);
    LDAModel model = lda.fit(dataSet);

    Matrix topics = model.topicsMatrix();

    // <editor-fold desc="Show topics">
    for (int topic = 0; topic < topicNum; topic++) {
      ArrayList<Tuple2<Double, String>> topicMap = new ArrayList<>();
      for (int word = 0; word < model.vocabSize(); word++) {
        topicMap
            .add(new Tuple2<java.lang.Double, String>(topics.apply(word, topic), vocabulary[word]));
      }

      JavaPairRDD pairRDD = new JavaSparkContext(session.sparkContext()).parallelizePairs(topicMap);
      JavaPairRDD sortedPairRDD = pairRDD.sortByKey(false);
      List<Tuple2<Integer, String>> sortedList = sortedPairRDD.collect();

      System.out.print("Topic " + topic + ":");

      int count = 0;
      for (Tuple2<Integer, String> tuple : sortedList) {
        System.out.print(tuple._2() + " ");
        if (count++ >= 2)
          break;
      }

      System.out.println();
    }
    // </editor-fold>

    try {
      model.write().overwrite().save(output);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override
  public void Predict(String comment, String model, CountVectorizerModel counterModel) {
    LocalLDAModel sameModel = LocalLDAModel.load(model);
    Matrix topics = sameModel.topicsMatrix();

    ConvertComment2Vector(comment, config.getProperty("Comment"), counterModel);
    Dataset<Row> dataset = session.read().format("libsvm").load(config.getProperty("Comment"));
    Dataset resultDataSet = sameModel.transform(dataset);
    resultDataSet.show();

    List result = resultDataSet.select("topicDistribution").collectAsList();
    String res = (result.get(0).toString().replace("[[", "").replace("]]", ""));
    String[] indexString = res.split(",");
    int indexMax = 0;
    double maxValue = 0.0;
    for (int i = 0; i < indexString.length; i++) {
      if ((Double) Double.parseDouble(indexString[i]) > maxValue) {
        maxValue = (Double) Double.parseDouble(indexString[i]);
        indexMax = i;
      }
    }

    int topic = indexMax;
    ArrayList<Tuple2<java.lang.Double, String>> topicMap = new ArrayList<>();
    for (int word = 0; word < sameModel.vocabSize(); word++) {
      topicMap
          .add(new Tuple2<java.lang.Double, String>(topics.apply(word, topic), vocabulary[word]));
    }

    JavaPairRDD pairRDD = new JavaSparkContext(session.sparkContext()).parallelizePairs(topicMap);
    JavaPairRDD sortedPairRDD = pairRDD.sortByKey(false);
    List<Tuple2<Integer, String>> sortedList = sortedPairRDD.collect();

    System.out.print("Topic " + topic + ":");
    int count = 0;
    for (Tuple2<Integer, String> tuple : sortedList) {
      System.out.print(tuple._2() + " ");
      if (count++ >= 2)
        break;
    }
    System.out.println();
    System.exit(0);
  }

  public enum tests {
    SEG, VEC, COM, BID, PRE
  }


}
