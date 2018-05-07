source /home1/r/romap/crf/crf_task/bin/activate
python /nlp/data/romap/law/task_3/models/exp.py

python /nlp/data/romap/law/task_3/models/evaluate.py concept lstm
python /nlp/data/romap/law/task_3/models/evaluate.py concept log
python /nlp/data/romap/law/task_3/models/evaluate.py concept svm

python /nlp/data/romap/law/task_3/models/evaluate.py google lstm
python /nlp/data/romap/law/task_3/models/evaluate.py google log
python /nlp/data/romap/law/task_3/models/evaluate.py google svm

python /nlp/data/romap/law/task_3/models/evaluate.py legal lstm
python /nlp/data/romap/law/task_3/models/evaluate.py legal log
python /nlp/data/romap/law/task_3/models/evaluate.py legal svm
