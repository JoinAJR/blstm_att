import tensorflow as tf
import numpy as np
import os
import datetime
import time

from att_lstm import AttLSTM
from bilstm_without_att import BiLSTM
from birnn_without_att import BiRNN
from lstm_without_att import LSTM
import data_helpers
import utils
from configure import FLAGS

from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# # 原始
# def train_and_eval():
#     with tf.device('/cpu:0'):
#         x_text, y, desc1, desc2 = data_helpers.load_data_and_labels(FLAGS.train_path)

#     # Build vocabulary
#     # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
#     # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
#     # =>
#     # [27 39 40 41 42  1 43  0  0 ... 0]
#     # dimension = FLAGS.max_sentence_length
#     vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
#     vacab_list = list(vocab_processor.fit_transform(x_text))

#     x = np.array(vacab_list)

#     print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
#     print("x = {0}".format(x.shape))
#     print("y = {0}".format(y.shape))
#     print("")


#     # Randomly shuffle data to split into train and test(dev)
#     np.random.seed(10)
#     shuffle_indices = np.random.permutation(np.arange(len(y)))
#     x_shuffled = x[shuffle_indices]
#     y_shuffled = y[shuffle_indices]

#     # x_type_shuffled = wType_np[shuffle_indices]

#     # Split train/test set
#     # TODO: This is very crude, should use cross-validation
#     dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
#     x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#     y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

#     # x_train_type, x_dev_type = x_type_shuffled[:dev_sample_index], x_type_shuffled[dev_sample_index:]
#     print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

#     with tf.Graph().as_default():
#         session_conf = tf.ConfigProto(
#             allow_soft_placement=FLAGS.allow_soft_placement,
#             log_device_placement=FLAGS.log_device_placement)
#         session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
#         sess = tf.Session(config=session_conf)
#         with sess.as_default():
#             # model = LSTM(
#             # model = BiRNN(
#             # model = BiLSTM(
#             model = AttLSTM(
#                 sequence_length=x_train.shape[1],
#                 num_classes=y_train.shape[1],
#                 vocab_size=len(vocab_processor.vocabulary_),
#                 embedding_size=FLAGS.embedding_dim,
#                 hidden_size=FLAGS.hidden_size,
#                 l2_reg_lambda=FLAGS.l2_reg_lambda)

#             # Define Training procedure
#             global_step = tf.Variable(0, name="global_step", trainable=False)
#             optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
#             gvs = optimizer.compute_gradients(model.loss)
#             capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
#             train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

#             # Output directory for models and summaries
#             timestamp = str(int(time.time()))
#             out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
#             print("Writing to {}\n".format(out_dir))

#             # Summaries for loss and accuracy
#             loss_summary = tf.summary.scalar("loss", model.loss)
#             acc_summary = tf.summary.scalar("accuracy", model.accuracy)

#             # Train Summaries
#             train_summary_op = tf.summary.merge([loss_summary, acc_summary])
#             train_summary_dir = os.path.join(out_dir, "summaries", "train")
#             train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

#             # Dev summaries
#             dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
#             dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
#             dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

#             # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
#             checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
#             checkpoint_prefix = os.path.join(checkpoint_dir, "model")
#             if not os.path.exists(checkpoint_dir):
#                 os.makedirs(checkpoint_dir)
#             saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

#             # Write vocabulary
#             vocab_processor.save(os.path.join(out_dir, "vocab"))

#             # Initialize all variables
#             sess.run(tf.global_variables_initializer())

#             # Pre-trained word2vec
#             if FLAGS.embedding_path:
#                 pretrain_W = utils.load_glove(FLAGS.embedding_path, FLAGS.embedding_dim, vocab_processor)
#                 sess.run(model.W_text.assign(pretrain_W))
#                 print("Success to load pre-trained word2vec model!\n")

#             # Generate batches
#             batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
#             # Training loop. For each batch...
#             best_f1 = 0.0  # For save checkpoint(model)
#             for batch in batches:
#                 x_batch, y_batch,  = zip(*batch)
#                 # Train
#                 feed_dict = {
#                     model.input_text: x_batch,
#                     model.input_y: y_batch,
#                     model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
#                     model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
#                     model.dropout_keep_prob: FLAGS.dropout_keep_prob
#                 }
#                 _, step, summaries, loss, accuracy = sess.run(
#                     [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
#                 train_summary_writer.add_summary(summaries, step)

#                 # Training log display
#                 if step % FLAGS.display_every == 0:
#                     time_str = datetime.datetime.now().isoformat()
#                     print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

#                 # Evaluation
#                 if step % FLAGS.evaluate_every == 0:
#                     print("\nEvaluation:")
#                     feed_dict = {
#                         model.input_text: x_dev,
#                         model.input_y: y_dev,
#                         model.emb_dropout_keep_prob: 1.0,
#                         model.rnn_dropout_keep_prob: 1.0,
#                         model.dropout_keep_prob: 1.0
#                     }
#                     summaries, loss, accuracy, predictions = sess.run(
#                         [dev_summary_op, model.loss, model.accuracy, model.predictions], feed_dict)
#                     dev_summary_writer.add_summary(summaries, step)

#                     time_str = datetime.datetime.now().isoformat()
#                     f1 = f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(6)), average="micro")
#                     print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
#                     print("[UNOFFICIAL] (2*9+1)-Way Macro-Average F1 Score : {:g}\n".format(f1))

#                     # Model checkpoint
#                     if best_f1 < f1:
#                         best_f1 = f1
#                         path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
#                         print("Saved model checkpoint to {}\n".format(path))
#     eval(checkpoint_dir)

# # 原始
# def eval(checkpoint_dir):
#     with tf.device('/gpu:0'):
#         x_text, y, desc1, desc2 = data_helpers.load_data_and_labels(FLAGS.test_path)

#     text_path = os.path.join(checkpoint_dir, "..", "vocab")
#     text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
#     vacab_list = list(text_vocab_processor.fit_transform(x_text))

#     x = np.array(vacab_list)

#     checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

#     graph = tf.Graph()
#     with graph.as_default():
#         session_conf = tf.ConfigProto(
#             allow_soft_placement=FLAGS.allow_soft_placement,
#             log_device_placement=FLAGS.log_device_placement)
#         session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
#         sess = tf.Session(config=session_conf)
#         with sess.as_default():
#             # Load the saved meta graph and restore variables
#             saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#             saver.restore(sess, checkpoint_file)

#             # Get the placeholders from the graph by name
#             input_text = graph.get_operation_by_name("input_text").outputs[0]
#             # input_y = graph.get_operation_by_name("input_y").outputs[0]
#             emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
#             rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
#             dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

#             # Tensors we want to evaluate
#             predictions = graph.get_operation_by_name("output/predictions").outputs[0]

#             # Generate batches for one epoch
#             batches = data_helpers.batch_iter(list(x), FLAGS.batch_size, 1, shuffle=False)

#             # Collect the predictions here
#             preds = []
#             for x_batch in batches:
#                 pred = sess.run(predictions, {input_text: x_batch,
#                                               emb_dropout_keep_prob: 1.0,
#                                               rnn_dropout_keep_prob: 1.0,
#                                               dropout_keep_prob: 1.0})
#                 preds.append(pred)
#             preds = np.concatenate(preds)
#             truths = np.argmax(y, axis=1)

#             print(truths)
#             result = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
#             # result = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
#             for i in range(len(preds)):
#                 result[truths[i]][preds[i]]+=1
#             print("===the prediction result===")
#             print("\t0\t1\t2\t3\t4\t5")
#             count = 0
#             for i in range(len(result)):
#                 print(str(count)+"\t"+str(result[i][0])+"\t"+str(result[i][1])+"\t"+str(result[i][2])+"\t"+str(result[i][3])+"\t"+str(result[i][4])+"\t"+str(result[i][5]))
#                 count+=1
#             precision = []
#             recall = []
#             for j in range(len(result)):
#                 p = round(result[j][j]/sum(result[j]),3)*100
#                 col = [x[j] for x in result]
#                 r = round(result[j][j]/sum(col),3)*100
#                 precision.append(p)
#                 recall.append(r)
#             f1_scores=[]
#             for k in range(len(precision)):
#                 if (precision[k]+recall[k]) == 0:
#                     f1_scores.append(0)
#                 else:
#                     f1 = round((2*precision[k]*recall[k])/(precision[k]+recall[k]),1)
#                     f1_scores.append(f1)
#             print(precision,recall,f1_scores)
#             relationName = ["before","after","simultaneous","include","be_included","vague"]
#             for l in range(6):
#                 print(relationName[l]+"acc:"+str(precision[l])+"%,recall:"+str(recall[l])+"%,f1:"+str(f1_scores[l])+"%")
#             precision_ave = round(sum(precision)/6,1)
#             recall_ave  = round(sum(recall)/6,1)
#             # f1_score_ave = round(sum(f1_scores)/6,1)
#             f1_score_ave = f1_score(truths, preds, labels=np.array(range(6)), average="micro")
#             print("acc_avg:"+str(precision_ave)+"%,recall_avg:"+str(recall_ave)+"%,f1:"+str(f1_score_ave)+"%")
#             print("modelFile:" + str(checkpoint_dir))



# #原始+refixFlag
# def eval(checkpoint_dir):
#     with tf.device('/gpu:0'):
#         x_text, y, refix_flag = data_helpers.load_data_and_labels_and_refixFlag(FLAGS.test_path,FLAGS.test_refix_path)

#     text_path = os.path.join(checkpoint_dir, "..", "vocab")
#     text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
#     vacab_list = list(text_vocab_processor.fit_transform(x_text))

#     x = np.array(vacab_list)

#     checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

#     graph = tf.Graph()
#     with graph.as_default():
#         session_conf = tf.ConfigProto(
#             allow_soft_placement=FLAGS.allow_soft_placement,
#             log_device_placement=FLAGS.log_device_placement)
#         session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
#         sess = tf.Session(config=session_conf)
#         with sess.as_default():
#             # Load the saved meta graph and restore variables
#             saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#             saver.restore(sess, checkpoint_file)

#             # Get the placeholders from the graph by name
#             input_text = graph.get_operation_by_name("input_text").outputs[0]
#             # input_y = graph.get_operation_by_name("input_y").outputs[0]
#             emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
#             rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
#             dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

#             # Tensors we want to evaluate
#             predictions = graph.get_operation_by_name("output/predictions").outputs[0]

#             # Generate batches for one epoch
#             batches = data_helpers.batch_iter(list(x), FLAGS.batch_size, 1, shuffle=False)

#             # Collect the predictions here
#             preds = []
#             for x_batch in batches:
#                 pred = sess.run(predictions, {input_text: x_batch,
#                                                 emb_dropout_keep_prob: 1.0,
#                                                 rnn_dropout_keep_prob: 1.0,
#                                                 dropout_keep_prob: 1.0})
#                 preds.append(pred)
#             preds = np.concatenate(preds)
#             truths = np.argmax(y, axis=1)
#             preds = refix(preds,truths,refix_flag)  

#             print(truths)
#             result = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
#             # result = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
#             for i in range(len(preds)):
#                 result[truths[i]][preds[i]] += 1
#             print("===the prediction result===")
#             print("\t0\t1\t2\t3\t4\t5")
#             count = 0
#             for i in range(len(result)):
#                 print(str(count) + "\t" + str(result[i][0]) + "\t" + str(result[i][1]) + "\t" + str(
#                     result[i][2]) + "\t" + str(result[i][3]) + "\t" + str(result[i][4]) + "\t" + str(
#                     result[i][5]))
#                 count += 1
#             precision = []
#             recall = []
#             for j in range(len(result)):
#                 p = round(result[j][j] / sum(result[j]), 3) * 100
#                 col = [x[j] for x in result]
#                 r = round(result[j][j] / sum(col), 3) * 100
#                 precision.append(p)
#                 recall.append(r)
#             f1_scores = []
#             for k in range(len(precision)):
#                 if (precision[k] + recall[k]) == 0:
#                     f1_scores.append(0)
#                 else:
#                     f1 = round((2 * precision[k] * recall[k]) / (precision[k] + recall[k]), 1)
#                     f1_scores.append(f1)
#             print(precision, recall, f1_scores)
#             relationName = ["before", "after", "simultaneous", "include", "be_included", "vague"]
#             for l in range(6):
#                 print(
#                     relationName[l] + "acc:" + str(precision[l]) + "%,recall:" + str(recall[l]) + "%,f1:" + str(
#                         f1_scores[l]) + "%")
#             precision_ave = round(sum(precision) / 6, 1)
#             recall_ave = round(sum(recall) / 6, 1)
#             # f1_score_ave = round(sum(f1_scores)/6,1)
#             f1_score_ave = f1_score(truths, preds, labels=np.array(range(6)), average="micro")
#             print("acc_avg:" + str(precision_ave) + "%,recall_avg:" + str(recall_ave) + "%,f1:" + str(
#                 f1_score_ave) + "%")
#             print("modelFile:" + str(checkpoint_dir))


# # 事件类型(句子补零后拼接特征)
# def train_and_eval():
#     with tf.device('/cpu:0'):
#         x_text, y, etypes = data_helpers.load_data_and_labels_and_etypes(FLAGS.train_path,FLAGS.train_etype_path)


#     # Build vocabulary
#     # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
#     # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
#     # =>
#     # [27 39 40 41 42  1 43  0  0 ... 0]
#     # dimension = FLAGS.max_sentence_length

#     vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
#     vacab_list = list(vocab_processor.fit_transform(x_text+etypes))

#     all_narray = np.array(vacab_list)
#     narray_len = len(all_narray)/2

#     text_np = all_narray[:int(narray_len)]
#     eType_np = all_narray[int(narray_len):]

#     eType_np = eType_np[:, 0:2]

#     # 矩阵拼接
#     x = tf.concat([text_np, eType_np],1)
#     # tensor转narray
#     x = tf.Session().run(x)

#     print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
#     print("x = {0}".format(x.shape))
#     print("y = {0}".format(y.shape))
#     print("")


#     # Randomly shuffle data to split into train and test(dev)
#     np.random.seed(10)
#     shuffle_indices = np.random.permutation(np.arange(len(y)))
#     x_shuffled = x[shuffle_indices]
#     y_shuffled = y[shuffle_indices]

#     # x_type_shuffled = eType_np[shuffle_indices]

#     # Split train/test set
#     # TODO: This is very crude, should use cross-validation
#     dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
#     x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#     y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

#     # x_train_etype, x_dev_etype = x_type_shuffled[:dev_sample_index], x_type_shuffled[dev_sample_index:]

#     print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

#     with tf.Graph().as_default():
#         session_conf = tf.ConfigProto(
#             allow_soft_placement=FLAGS.allow_soft_placement,
#             log_device_placement=FLAGS.log_device_placement)
#         session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
#         sess = tf.Session(config=session_conf)
#         with sess.as_default():
#             # model = LSTM(
#             # model = BiRNN(
#             # model = BiLSTM(
#             model = AttLSTM(
#                 sequence_length=x_train.shape[1],
#                 num_classes=y_train.shape[1],
#                 vocab_size=len(vocab_processor.vocabulary_),
#                 embedding_size=FLAGS.embedding_dim,
#                 hidden_size=FLAGS.hidden_size,
#                 l2_reg_lambda=FLAGS.l2_reg_lambda)

#             # Define Training procedure
#             global_step = tf.Variable(0, name="global_step", trainable=False)
#             optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
#             gvs = optimizer.compute_gradients(model.loss)
#             capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
#             train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

#             # Output directory for models and summaries
#             timestamp = str(int(time.time()))
#             out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
#             print("Writing to {}\n".format(out_dir))

#             # Summaries for loss and accuracy
#             loss_summary = tf.summary.scalar("loss", model.loss)
#             acc_summary = tf.summary.scalar("accuracy", model.accuracy)

#             # Train Summaries
#             train_summary_op = tf.summary.merge([loss_summary, acc_summary])
#             train_summary_dir = os.path.join(out_dir, "summaries", "train")
#             train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

#             # Dev summaries
#             dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
#             dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
#             dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

#             # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
#             checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
#             checkpoint_prefix = os.path.join(checkpoint_dir, "model")
#             if not os.path.exists(checkpoint_dir):
#                 os.makedirs(checkpoint_dir)
#             saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

#             # Write vocabulary
#             vocab_processor.save(os.path.join(out_dir, "vocab"))

#             # Initialize all variables
#             sess.run(tf.global_variables_initializer())

#             # Pre-trained word2vec
#             if FLAGS.embedding_path:
#                 pretrain_W = utils.load_glove(FLAGS.embedding_path, FLAGS.embedding_dim, vocab_processor)
#                 sess.run(model.W_text.assign(pretrain_W))
#                 print("Success to load pre-trained word2vec model!\n")

#             # Generate batches
#             batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
#             # Training loop. For each batch...
#             best_f1 = 0.0  # For save checkpoint(model)
#             for batch in batches:
#                 x_batch, y_batch = zip(*batch)
#                 # x_batch, y_batch, x_train_etype  = zip(*batch)
#                 # Train
#                 feed_dict = {
#                     model.input_text: x_batch,
#                     # model.input_etype:x_train_etype,
#                     model.input_y: y_batch,
#                     model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
#                     model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
#                     model.dropout_keep_prob: FLAGS.dropout_keep_prob
#                 }
#                 _, step, summaries, loss, accuracy = sess.run(
#                     [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
#                 train_summary_writer.add_summary(summaries, step)

#                 # Training log display
#                 if step % FLAGS.display_every == 0:
#                     time_str = datetime.datetime.now().isoformat()
#                     print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

#                 # Evaluation
#                 if step % FLAGS.evaluate_every == 0:
#                     print("\nEvaluation:")
#                     feed_dict = {
#                         model.input_text: x_dev,
#                         model.input_y: y_dev,
#                         # model.input_etype:x_dev_etype,
#                         model.emb_dropout_keep_prob: 1.0,
#                         model.rnn_dropout_keep_prob: 1.0,
#                         model.dropout_keep_prob: 1.0
#                     }
#                     summaries, loss, accuracy, predictions = sess.run(
#                         [dev_summary_op, model.loss, model.accuracy, model.predictions], feed_dict)
#                     dev_summary_writer.add_summary(summaries, step)

#                     time_str = datetime.datetime.now().isoformat()
#                     f1 = f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(6)), average="micro")
#                     print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
#                     print("[UNOFFICIAL] (2*9+1)-Way Macro-Average F1 Score : {:g}\n".format(f1))

#                     # Model checkpoint
#                     if best_f1 < f1:
#                         best_f1 = f1
#                         path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
#                         print("Saved model checkpoint to {}\n".format(path))

#     eval(checkpoint_dir)
# # 事件类型(句子补零后拼接特征)
# def eval(checkpoint_dir):
#     with tf.device('/gpu:0'):
#         x_text, y, etypes = data_helpers.load_data_and_labels_and_etypes(FLAGS.test_path, FLAGS.test_etype_path)

#     text_path = os.path.join(checkpoint_dir, "..", "vocab")
#     text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
#     vacab_list = list(text_vocab_processor.fit_transform(x_text + etypes))

#     all_narray = np.array(vacab_list)
#     narray_len = len(all_narray) / 2

#     text_np = all_narray[:int(narray_len)]
#     eType_np = all_narray[int(narray_len):]

#     eType_np = eType_np[:, 0:2]

#     # 矩阵拼接
#     x = tf.concat([text_np, eType_np], 1)
#     # tensor转narray
#     x = tf.Session().run(x)

#     checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

#     graph = tf.Graph()
#     with graph.as_default():
#         session_conf = tf.ConfigProto(
#             allow_soft_placement=FLAGS.allow_soft_placement,
#             log_device_placement=FLAGS.log_device_placement)
#         session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
#         sess = tf.Session(config=session_conf)
#         with sess.as_default():
#             # Load the saved meta graph and restore variables
#             saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#             saver.restore(sess, checkpoint_file)

#             # Get the placeholders from the graph by name
#             input_text = graph.get_operation_by_name("input_text").outputs[0]
#             # input_y = graph.get_operation_by_name("input_y").outputs[0]
#             emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
#             rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
#             dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

#             # Tensors we want to evaluate
#             predictions = graph.get_operation_by_name("output/predictions").outputs[0]

#             # Generate batches for one epoch
#             batches = data_helpers.batch_iter(list(x), FLAGS.batch_size, 1, shuffle=False)

#             # Collect the predictions here
#             preds = []
#             for x_batch in batches:
#                 pred = sess.run(predictions, {input_text: x_batch,
#                                               emb_dropout_keep_prob: 1.0,
#                                               rnn_dropout_keep_prob: 1.0,
#                                               dropout_keep_prob: 1.0})
#                 preds.append(pred)
#             preds = np.concatenate(preds)
#             truths = np.argmax(y, axis=1)

#             print(truths)
#             result = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
#             # result = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
#             for i in range(len(preds)):
#                 result[truths[i]][preds[i]]+=1
#             print("===the prediction result===")
#             print("\t0\t1\t2\t3\t4\t5")
#             count = 0
#             for i in range(len(result)):
#                 print(str(count)+"\t"+str(result[i][0])+"\t"+str(result[i][1])+"\t"+str(result[i][2])+"\t"+str(result[i][3])+"\t"+str(result[i][4])+"\t"+str(result[i][5]))
#                 count+=1
#             precision = []
#             recall = []
#             for j in range(len(result)):
#                 p = round(result[j][j]/sum(result[j]),3)*100
#                 col = [x[j] for x in result]
#                 r = round(result[j][j]/sum(col),3)*100
#                 precision.append(p)
#                 recall.append(r)
#             f1_scores=[]
#             for k in range(len(precision)):
#                 if (precision[k]+recall[k]) == 0:
#                     f1_scores.append(0)
#                 else:
#                     f1 = round((2*precision[k]*recall[k])/(precision[k]+recall[k]),1)
#                     f1_scores.append(f1)
#             print(precision,recall,f1_scores)
#             relationName = ["before","after","simultaneous","include","be_included","vague"]
#             for l in range(6):
#                 print(relationName[l]+"acc:"+str(precision[l])+"%,recall:"+str(recall[l])+"%,f1:"+str(f1_scores[l])+"%")
#             precision_ave = round(sum(precision)/6,1)
#             recall_ave  = round(sum(recall)/6,1)
#             # f1_score_ave = round(sum(f1_scores)/6,1)
#             f1_score_ave = f1_score(truths, preds, labels=np.array(range(6)), average="micro")
#             print("acc_avg:"+str(precision_ave)+"%,recall_avg:"+str(recall_ave)+"%,f1:"+str(f1_score_ave)+"%")
#             print("modelFile:" + str(checkpoint_dir))

# 事件类型(句子补零前拼接特征)
def train_and_eval():
    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_dataWithetypes_and_labels(FLAGS.train_path,FLAGS.train_etype_path)


    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = FLAGS.max_sentence_length

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    vacab_list = list(vocab_processor.fit_transform(x_text))

    x = np.array(vacab_list)

    print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")


    # Randomly shuffle data to split into train and test(dev)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # x_type_shuffled = eType_np[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    # x_train_etype, x_dev_etype = x_type_shuffled[:dev_sample_index], x_type_shuffled[dev_sample_index:]

    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # model = LSTM(
            # model = BiRNN(
            # model = BiLSTM(
            model = AttLSTM(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(model.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.embedding_path:
                pretrain_W = utils.load_glove(FLAGS.embedding_path, FLAGS.embedding_dim, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # x_batch, y_batch, x_train_etype  = zip(*batch)
                # Train
                feed_dict = {
                    model.input_text: x_batch,
                    # model.input_etype:x_train_etype,
                    model.input_y: y_batch,
                    model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                    model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict = {
                        model.input_text: x_dev,
                        model.input_y: y_dev,
                        # model.input_etype:x_dev_etype,
                        model.emb_dropout_keep_prob: 1.0,
                        model.rnn_dropout_keep_prob: 1.0,
                        model.dropout_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, predictions = sess.run(
                        [dev_summary_op, model.loss, model.accuracy, model.predictions], feed_dict)
                    dev_summary_writer.add_summary(summaries, step)

                    time_str = datetime.datetime.now().isoformat()
                    f1 = f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(6)), average="micro")
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    print("[UNOFFICIAL] (2*9+1)-Way Macro-Average F1 Score : {:g}\n".format(f1))

                    # Model checkpoint
                    if best_f1 < f1:
                        best_f1 = f1
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))

    eval(checkpoint_dir)
# 事件类型(句子补零前拼接特征)
def eval(checkpoint_dir):
    with tf.device('/gpu:0'):
        x_text, y = data_helpers.load_dataWithetypes_and_labels(FLAGS.test_path, FLAGS.test_etype_path)

    text_path = os.path.join(checkpoint_dir, "..", "vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    vacab_list = list(text_vocab_processor.fit_transform(x_text))

    x = np.array(vacab_list)

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
            rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            preds = []
            for x_batch in batches:
                pred = sess.run(predictions, {input_text: x_batch,
                                              emb_dropout_keep_prob: 1.0,
                                              rnn_dropout_keep_prob: 1.0,
                                              dropout_keep_prob: 1.0})
                preds.append(pred)
            preds = np.concatenate(preds)
            truths = np.argmax(y, axis=1)

            print(truths)
            result = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
            # result = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            for i in range(len(preds)):
                result[truths[i]][preds[i]]+=1
            print("===the prediction result===")
            print("\t0\t1\t2\t3\t4\t5")
            count = 0
            for i in range(len(result)):
                print(str(count)+"\t"+str(result[i][0])+"\t"+str(result[i][1])+"\t"+str(result[i][2])+"\t"+str(result[i][3])+"\t"+str(result[i][4])+"\t"+str(result[i][5]))
                count+=1
            precision = []
            recall = []
            for j in range(len(result)):
                p = round(result[j][j]/sum(result[j]),3)*100
                col = [x[j] for x in result]
                r = round(result[j][j]/sum(col),3)*100
                precision.append(p)
                recall.append(r)
            f1_scores=[]
            for k in range(len(precision)):
                if (precision[k]+recall[k]) == 0:
                    f1_scores.append(0)
                else:
                    f1 = round((2*precision[k]*recall[k])/(precision[k]+recall[k]),1)
                    f1_scores.append(f1)
            print(precision,recall,f1_scores)
            relationName = ["before","after","simultaneous","include","be_included","vague"]
            for l in range(6):
                print(relationName[l]+"acc:"+str(precision[l])+"%,recall:"+str(recall[l])+"%,f1:"+str(f1_scores[l])+"%")
            precision_ave = round(sum(precision)/6,1)
            recall_ave  = round(sum(recall)/6,1)
            # f1_score_ave = round(sum(f1_scores)/6,1)
            f1_score_ave = f1_score(truths, preds, labels=np.array(range(6)), average="micro")
            print("acc_avg:"+str(precision_ave)+"%,recall_avg:"+str(recall_ave)+"%,f1:"+str(f1_score_ave)+"%")
            print("modelFile:" + str(checkpoint_dir))


def refix(preds,truths,refix_flag):
    for i in range(len(preds)):
        if refix_flag[i] == "true":
            preds[i] = truths[i]
    return preds


def main(_):
    train_and_eval()


if __name__ == "__main__":
    tf.app.run()
