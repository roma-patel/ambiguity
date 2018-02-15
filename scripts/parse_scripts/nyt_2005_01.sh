source /home1/r/romap/crf/crf_task/bin/activate
cd /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09
java -Xmx20g -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,parse -file /nlp/data/romap/ambig/docs/nyt_2005_01.txt -outputDirectory /nlp/data/romap/ambig/parse/