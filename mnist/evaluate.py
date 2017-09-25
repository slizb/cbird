
import time
import os
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from datetime import datetime


scores = []

# todo: refactor this code
class ModelEvaluator:

    def retrieve_closest_elements(test_code, test_label, learned_codes):
        distances = []
        for code in learned_codes:
            distance = np.linalg.norm(code - test_code)
            distances.append(distance)
        nb_elements = learned_codes.shape[0]
        distances = np.array(distances)
        learned_code_index = np.arange(nb_elements)
        labels = np.copy(y_train).astype('float32')
        labels[labels != test_label] = -1
        labels[labels == test_label] = 1
        labels[labels == -1] = 0
        distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
        sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

        sorted_distances = 28 - sorted_distance_with_labels[:, 0]
        sorted_labels = sorted_distance_with_labels[:, 1]
        sorted_indexes = sorted_distance_with_labels[:, 2]
        return sorted_distances, sorted_labels, sorted_indexes

    def compute_average_precision_score(test_codes, test_labels, learned_codes, n_samples, dir_name):
        out_labels = []
        out_distances = []
        retrieved_elements_indexes = []
        for i in range(len(test_codes)):
            sorted_distances, sorted_labels, sorted_indexes = retrieve_closest_elements(test_codes[i],
                                                                                        test_labels[i],
                                                                                        learned_codes)
            out_distances.append(sorted_distances[:n_samples])
            out_labels.append(sorted_labels[:n_samples])
            retrieved_elements_indexes.append(sorted_indexes[:n_samples])

        out_labels = np.array(out_labels)
        out_labels_file_name = dir_name + '/out_labels_{}'.format(n_samples)
        np.save(out_labels_file_name, out_labels)

        out_distances_file_name = dir_name + '/out_distances_{}'.format(n_samples)
        out_distances = np.array(out_distances)
        np.save(out_distances_file_name, out_distances)
        score = label_ranking_average_precision_score(out_labels, out_distances)
        scores.append(score)
        return score

    def evaluate_model(n_train_samples=[10,50,100,200,300,400,500,750,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,60000],
                       n_test_samples=1000):
        # pass in unique date-time-based sub_dir
        tmstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_nm = 'computed_data/' + tmstamp
        os.mkdir(dir_nm)
        for n_train_sample in n_train_samples:
            test_model(n_test_samples, n_train_sample, dir_nm)

        np.save(dir_nm + '/scores', np.array(scores))
        np.save(dir_nm + '/n_samples', np.array(n_train_sample))

    def test_model(n_test_samples, n_train_samples, dir_name):
        learned_codes = encoder.predict(x_train)
        learned_codes = learned_codes.reshape(learned_codes.shape[0],
                                              learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[
                                                  3])
        test_codes = encoder.predict(x_test)
        test_codes = test_codes.reshape(test_codes.shape[0],
                                        test_codes.shape[1] * test_codes.shape[2] * test_codes.shape[3])
        indexes = np.arange(len(y_test))
        np.random.shuffle(indexes)
        indexes = indexes[:n_test_samples]

        print('Start computing score for {} train samples'.format(n_train_samples))
        t1 = time.time()
        score = compute_average_precision_score(test_codes[indexes],
                                                y_test[indexes],
                                                learned_codes,
                                                n_train_samples,
                                                dir_name)
        t2 = time.time()
        print('Score computed in: ', t2 - t1)
        print('Model score:', score)

