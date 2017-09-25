
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from scipy import integrate


def plot_performance_curve(base_dir, results_dir):
    def assign_dir(x):
        path_elements = [base_dir, results_dir, x]
        return os.path.join(*path_elements)

    performance = np.load(assign_dir('scores.npy'))
    n = np.load(assign_dir('n_samples.npy'))
    # compute area under performance curve
    model_score = integrate.simps(x=n, y=performance) / np.max(n)
    model_score = round(model_score, 4)
    plt.plot(n, performance, label=results_dir + ': ' + str(model_score))
    plt.legend()
    plt.xlabel('# Retrieved')
    plt.ylabel('Precision')
    plt.savefig(assign_dir('performance_curve.png'),
                bbox_inches='tight',
                dpi=200)
    plt.show()


def plot_denoised_images(noisy, autoencoder):
    denoised_images = autoencoder.predict(noisy.reshape(noisy.shape[0],
                                                        noisy.shape[1],
                                                        noisy.shape[2],
                                                        1))
    test_img = noisy[0]
    resized_test_img = cv2.resize(test_img, (280, 280))
    cv2.imshow('input', resized_test_img)
    cv2.waitKey(0)
    output = denoised_images[0]
    resized_output = cv2.resize(output, (280, 280))
    cv2.imshow('output', resized_output)
    cv2.waitKey(0)
    cv2.imwrite('test_results/noisy_image.jpg', 255 * resized_test_img)
    cv2.imwrite('test_results/denoised_image.jpg', 255 * resized_output)
