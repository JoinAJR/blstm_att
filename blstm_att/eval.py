import tensorflow as tf
import numpy as np
import os
import subprocess

import data_helpers
import utils
from configure import FLAGS
from sklearn.metrics import f1_score


def eval():
    with tf.device('/gpu:0'):
        x_text, y, desc1, desc2, wType, type_index = data_helpers.load_data_and_labels(FLAGS.test_path)

    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    x = np.array(list(text_vocab_processor.transform(x_text)))
    
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

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
            print("modelFile:" + str(FLAGS.checkpoint_dir))


def evalNewData():
    with tf.device('/gpu:0'):
        x_text= data_helpers.load_data(FLAGS.test_path)

    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    x = np.array(list(text_vocab_processor.transform(x_text)))

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

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
            with open(os.path.join("predict_result","1_300_model_1572577057_1105.txt"), 'a', encoding="utf-8") as resultFile:
                for i in range(0,len(x_text)):
                    resultFile.write(x_text[i] + "\n")
                    resultFile.write(str(preds[i]+1) + "\n\n")

def main(_):
    eval()
    # evalNewData()

if __name__ == "__main__":
    tf.app.run()