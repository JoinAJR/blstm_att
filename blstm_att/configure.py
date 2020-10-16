import argparse
import sys

# 设置各项参数：训练集路径，测试集路径，最大句子长度，用于验证的训练数据百分比
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    # Data loading params
    parser.add_argument("--train_path",
                        default="clinic_data/train_has_ID_all_1_1000(new)_forEtypeTrain_text_v1.1.txt",
                        type=str, help="Path of train data")  # default del_1960_train.txt
    parser.add_argument("--test_path", default="clinic_data/test_has_ID_all(new)_forEtypeTest_text_v1.1.txt",
                        type=str, help="Path of test data")  # default del_840_test.txt #our_test.txt
    parser.add_argument("--max_sentence_length", default=300,
                        type=int, help="Max sentence length in data")
    parser.add_argument("--dev_sample_percentage", default=0.1,
                        type=float, help="Percentage of the training data to use for validation")

    parser.add_argument("--train_etype_path", default="clinic_data/train_has_ID_all_1_1000(new)_forEtypeTrain_etype_v1.1.txt",
                        type=str, help="Path of train_etype data")  # default del_840_test.txt #our_test.txt
    parser.add_argument("--test_etype_path", default="clinic_data/test_has_ID_all(new)_forEtypeTest_etype_v1.1.txt",
                        type=str, help="Path of train_etype data")  # default del_840_test.txt #our_test.txt


    # # # Data loading params
    # parser.add_argument("--train_path", default="SemEval2010_task8_all_data/SemEval2010_task8_training/del_1960_train.txt",
    #                     type=str, help="Path of train data")#default del_1960_train.txt
    # parser.add_argument("--test_path", default="SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/our_test.txt",
    #                     type=str, help="Path of test data")#default del_840_test.txt #our_test.txt
    # parser.add_argument("--max_sentence_length", default=65,
    #                     type=int, help="Max sentence length in data")
    # parser.add_argument("--dev_sample_percentage", default=0.1,
    #                     type=float, help="Percentage of the training data to use for validation")

    # Model Hyper-parameters
    # Embeddings
    parser.add_argument("--embedding_path", default="/wordvector/cc.zh.300.vec",
                        type=str, help="Path of pre-trained word embeddings (glove)")
    # parser.add_argument("--embedding_path", default="",
    #                     type=str, help="Path of pre-trained word embeddings (glove)")
    parser.add_argument("--embedding_dim", default=300,
                        type=int, help="Dimensionality of word embedding (default: 100)")
    parser.add_argument("--emb_dropout_keep_prob", default=0.7,
                        type=float, help="Dropout keep probability of embedding layer (default: 0.7)")
    # AttLSTM
    parser.add_argument("--hidden_size", default=100,
                        type=int, help="Dimensionality of RNN hidden (default: 100)")
    parser.add_argument("--rnn_dropout_keep_prob", default=0.7,
                        type=float, help="Dropout keep probability of RNN (default: 0.7)")

    # Misc
    parser.add_argument("--desc", default="",
                        type=str, help="Description for model")
    parser.add_argument("--dropout_keep_prob", default=0.5,
                        type=float, help="Dropout keep probability of output layer (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", default=1e-5,
                        type=float, help="L2 regularization lambda (default: 1e-5)")

    # Training parameters
    parser.add_argument("--batch_size", default=32,
                        type=int, help="Batch Size (default: 10)")
    parser.add_argument("--num_epochs", default=60,
                        type=int, help="Number of training epochs (Default: 100)")
    parser.add_argument("--display_every", default=10,
                        type=int, help="Number of iterations to display training information")
    parser.add_argument("--evaluate_every", default=100,
                        type=int, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", default=5,
                        type=int, help="Number of checkpoints to store (default: 5)")
    parser.add_argument("--learning_rate", default=1.0,
                        type=float, help="Which learning rate to start with (Default: 1.0)")
    parser.add_argument("--decay_rate", default=0.9,
                        type=float, help="Decay rate for learning rate (Default: 0.9)")

    # Testing parameters
    parser.add_argument("--checkpoint_dir",
                        default="C:/joinajr/research_code/REL/凌佳君大数据服务器代码及训练模型/凌佳君/Attention-Based-BiLSTM-relation-extraction_ljj/runs/1599913041/checkpoints",
                        type=str, help="Checkpoint directory from training run")

        # Misc Parameters
    parser.add_argument("--allow_soft_placement", default=True,
                        type=bool, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False,
                        type=bool, help="Log placement of ops on devices")
    parser.add_argument("--gpu_allow_growth", default=True,
                        type=bool, help="Allow gpu memory growth")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args


FLAGS = parse_args()
