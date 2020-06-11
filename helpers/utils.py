import numpy as np
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical


def top_k_accuracy(y_true, y_pred, k=1):
    """From: https://github.com/chainer/chainer/issues/606
    Expects both y_true and y_pred to be one-hot encoded.
    """
    argsorted_y = np.argsort(y_pred)[:, -k:]
    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()*100


def calculate_error(y_true, y_pred):
    return np.sum([y_true != y_pred]) / len(y_true)


def calculate_confidence(y_true, y_pred_prob, bin_size=0.1, min_samples=20):
    bins = np.arange(0, 1, bin_size)
    y_true = np.argmax(y_true, axis=-1)
    confidences = np.amax(y_pred_prob, axis=-1)
    y_pred_prob_c = np.argmax(y_pred_prob, axis=-1)
    accuracies = np.zeros((len(bins, )))
    confs = np.zeros((len(bins, )))
    ece = 0
    errors = list()
    for index, confidence_threshold in enumerate(bins):
        filtered = np.where((confidence_threshold < confidences) & (confidences <= confidence_threshold + bin_size))[0]
        if len(filtered) > min_samples:
            accuracies[index] = np.mean(1 - calculate_error(y_true[filtered], y_pred_prob_c[filtered]))
            confs[index] = np.mean(confidences[filtered])
            error = np.abs(accuracies[index] - confs[index])
            ece += len(filtered) / len(y_true) * error * 100
            errors.append(error)
    mce = np.amax(errors) * 100
    return confs, accuracies, ece, mce


def plot_confidence(confidence, accuracy, ax, bin_size=0.1):
    confidence_thresholds = np.arange(0, 1 + bin_size, bin_size)
    x = list()
    y = list()
    diff = list()
    fc = (1, 0, 0, 0.1)
    ec = (1, 0, 0, 1)
    fc2 = (65 / 255., 105 / 255., 1., 0.8)
    fc = (220/255, 20/255, 60/255, 0.1)
    ec = (220/255, 20/255, 60/255, 1)
    fc2 = (0.8, 0.8, 0.8, 0.5)
    for confidence_threshold in confidence_thresholds:
        filtered = np.where((confidence > confidence_threshold) & (confidence <= confidence_threshold + bin_size))
        x.append(np.mean(confidence[filtered]))
        y.append(np.mean(accuracy[filtered]))
        diff.append(np.mean(accuracy[filtered]) - max(0, np.mean(accuracy[filtered]) - np.mean(confidence[filtered])))
    bs = ax.bar(confidence_thresholds + bin_size / 2, x, fc=fc, ec=ec, width=0.1, lw=1,
                label='Gap')
    for b in bs:
        b.set_hatch('//')
    bs = ax.bar(confidence_thresholds + bin_size / 2, y, fc=fc2, ec=ec,
                width=0.1, lw=2)
    for b in bs:
        b.set_hatch('//')
    ax.bar(confidence_thresholds + bin_size / 2, diff, fc=fc2, ec='k', width=0.1, alpha=1, lw=1, label='Outputs')
    ax.legend(loc=2)
    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'k--', )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def np_dice_coef(y_true, y_pred, threshold=0.5):
    if np.amax(y_pred) > 1 or np.amax(y_true) > 1:
        print('warning: values must be between 0 and 1!')
    smooth = 1.
    y_th = np.copy(y_pred)
    y_th[y_th >= threshold] = 1
    y_th[y_th < threshold] = 0
    y_true_f = y_true.flatten()
    y_pred_f = y_th.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def gnll(y_true, y_mean, y_var):
    y_diff = y_true - y_mean
    division = np.square(y_diff) / y_var
    gnll = 0.5 * np.mean(np.log(y_var)) + 0.5 * np.mean(division) + 0.5 * np.log(2 * np.pi)
    return gnll

def brier_multi(targets, probs):
    return np.mean(np.sum((probs - targets) ** 2, axis=1))


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
