import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score


def classification_error(y_true, y_pred):
    return 100 * (1 - np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1)) / np.sum(y_true))


class CalibrationErrors:
    def __init__(self, y_true, y_pred_prob, bin_size, min_samples, plot_with_counts=False):
        self.y_true = y_true
        self.y_pred_prob = y_pred_prob
        self.bin_size = bin_size
        self.min_samples = min_samples
        self.plot_with_counts = plot_with_counts

    def calculate_calibration_errors(self):
        bins = np.arange(0, 1, self.bin_size)
        # y_true = np.argmax(self.y_true, axis=-1)
        confidences = np.amax(self.y_pred_prob, axis=-1)
        y_pred_prob_c = np.argmax(self.y_pred_prob, axis=-1)
        y_pred_prob_c = to_categorical(y_pred_prob_c, num_classes=np.shape(self.y_pred_prob)[-1])
        accuracies = np.zeros((len(bins, )))
        confs = np.zeros((len(bins, )))
        ece = 0
        errors = list()
        counts = list()

        for index, confidence_threshold in enumerate(bins):
            filtered = np.where((confidence_threshold < confidences) &
                                (confidences <= confidence_threshold + self.bin_size))[0]
            if len(filtered) > self.min_samples:
                accuracies[index] = accuracy_score(self.y_true[filtered], y_pred_prob_c[filtered])
                confs[index] = np.mean(confidences[filtered])
                error = np.abs(accuracies[index] - confs[index])
                ece += len(filtered) / len(self.y_true) * error * 100
                errors.append(error)
            counts.append(len(filtered))
        mce = np.amax(errors) * 100
        return confs, accuracies, ece, mce, counts

    def plot_reliability_diagram(self, ax):
        confidence, accuracy, _, _, counts = self.calculate_calibration_errors()
        confidence_thresholds = np.arange(0, 1 + self.bin_size, self.bin_size)
        x = list()
        y = list()
        diff = list()
        fc = (220 / 255, 20 / 255, 60 / 255, 0.2)
        ec = (220 / 255, 20 / 255, 60 / 255, 1)
        fc2 = (0.75, 0.75, 0.75, 1)
        for confidence_threshold in confidence_thresholds:
            filtered = np.where((confidence > confidence_threshold) &
                                (confidence <= confidence_threshold + self.bin_size))
            x.append(np.mean(confidence[filtered]))
            y.append(np.mean(accuracy[filtered]))
            diff.append(
                np.mean(accuracy[filtered]) - max(0, np.mean(accuracy[filtered]) - np.mean(confidence[filtered])))
        bs1 = ax.bar(confidence_thresholds + self.bin_size / 2, x, fc=fc, ec=ec, width=0.1, lw=1,
                     label='Gap')
        for b in bs1:
            b.set_hatch('//')
        bs2 = ax.bar(confidence_thresholds + self.bin_size / 2, y, fc=fc2, ec=ec,
                     width=0.1, lw=2)
        for b in bs2:
            b.set_hatch('//')
        ax.bar(confidence_thresholds + self.bin_size / 2, diff, fc=fc2, ec='k', width=0.1, alpha=1, lw=1,
               label='Outputs')
        ax.legend(loc=2, fontsize=15)
        ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'k--', )
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)

        if self.plot_with_counts:
            for index, confidence_threshold in enumerate(confidence_thresholds):
                if not np.isnan(diff[index]):
                    ax.text(confidence_threshold + self.bin_size / 3, diff[index] - 0.03,
                            str(counts[index] // 1000) + 'k',
                            fontsize=5)


def average_binary_entropy(pred_positive):
    pred_positive[pred_positive < 0.01] = 0.001
    pred_positive[pred_positive > 0.99] = 0.999
    pred_negative = 1 - pred_positive
    average_entropy = -np.sum(pred_positive * np.log(pred_positive) +
                              pred_negative * np.log(pred_negative)) / len(pred_positive)
    return average_entropy


def brier_multi(targets, probs):
    return np.mean(np.sum((probs - targets) ** 2, axis=1))
