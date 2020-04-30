from __future__ import absolute_import
import sys
from ctr.algorithm import fctl
from ctr.common import utility

def main(min_loss, max_epoch, model_dir, output_dir ):
    train_fs = utility.FeatureStream(utility.get_date_file_path("app_feature_map_v2.json"),
    utility.get_date_file_path("app_train_features_v2.csv"))
    print ("training......")
    alg = fctl.Fctl(train_fs.feature_count())

    logloss_list = []
    avgloss_list = []
    precision_list = []
    recall_list = []

    logloss_list_test = []
    avgloss_list_test = []
    precision_list_test = []
    recall_list_test = []

    epoch = 0
    valid_avglogloss = 1

    epoch_list = []

    while valid_avglogloss > min_loss and epoch < max_epoch:
        if epoch == 0:
            logloss, avglogloss, precision, recall = alg.train(train_fs, 0.1, 1, 0.05, 0.01, False)
        else:
            alg.load_model(os.path.join(model_dir, 'app_model_'+str(epoch-1)+'.txt'))

            train_fs = utility.FeatureStream(utility.get_date_file_path("app_feature_map_v2.json"),
                                         utility.get_date_file_path("app_train_features_v2.csv"))
            logloss, avglogloss, precision, recall = alg.train(train_fs, 0.1, 1, 0.05, 0.01, True)

        epoch_list.append(epoch)
        logloss_list.append(logloss)
        avgloss_list.append(avglogloss)
        precision_list.append(precision)
        recall_list.append(recall)

        alg.dump_model(os.path.join(model_dir, 'app_model_'+str(epoch)+'.txt'))
        print(epoch)
        if epoch%2==0:
            print('validation...')
            alg.load_model(utility.get_date_file_path(os.path.join(model_dir, "app_model_" + str(epoch) + '.txt'))
            test_fs = utility.FeatureStream(utility.get_date_file_path("app_feature_map_v2.json"),
                                            utility.get_date_file_path("app_test_features_v2.csv"))
            valid_logloss, valid_avglogloss, valid_precision, valid_recall = alg.test(test_fs)

        logloss_list_test.append(valid_logloss)
        avgloss_list_test.append(valid_avglogloss)
        precision_list_test.append(valid_precision)
        recall_list_test.append(valid_recall)
        epoch += 1

    df = pd.DataFrame()
    df['logloss'] = logloss_list
    df['avglogloss'] = avgloss_list
    df['precision'] = precision_list
    df['recall'] = recall_list
    df_valid = pd.DataFrame()
    df_valid['logloss'] = logloss_list_test
    df_valid['avglogloss'] = avgloss_list_test
    df_valid['precision'] = precision_list_test
    df_valid['recall'] = recall_list_test
    df.to_csv(os.path.join(output_dir, 'training.csv'), index=False)
    df_valid.to_csv(os.path.join(output_dir,'validation.csv'), index=False)


if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__),'model_result')
    output_dir = os.path.join(os.path.dirname(__file__),'output_dir')
    main(0.0001, 1000, model_dir, output_dir)
