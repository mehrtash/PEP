import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from helpers.utils import top_k_accuracy, calculate_confidence, plot_confidence, calculate_error, brier_multi, \
    CalibrationErrors
from tqdm import tqdm


def stat_sig_nll(y_true, y_pred_1, y_pred_2, N_BOOTSTRAP=100):
    observed_differences = log_loss(y_true, y_pred_1) - log_loss(y_true, y_pred_2)
    nlls = list()
    for i in tqdm(range(N_BOOTSTRAP)):
        idx_data_1 = np.random.randint(len(y_pred_1), size=len(y_pred_1))
        idx_data_2 = np.random.randint(len(y_pred_2), size=len(y_pred_2))
        frames1 = y_pred_1[idx_data_1, :]
        y_true1 = y_true[idx_data_1, :]
        frames2 = y_pred_2[idx_data_2, :]
        y_true2 = y_true[idx_data_2, :]
        frames = np.concatenate([frames1, frames2], axis=0)
        y_true_goh = np.concatenate([y_true1, y_true2], axis=0)
        idx_data = np.random.randint(len(y_true_goh), size=len(y_true_goh))
        #
        frames = frames[idx_data]
        y_true_goh = y_true_goh[idx_data]
        a = frames[:len(frames) // 2]
        b = frames[len(frames) // 2:]
        g_a = y_true_goh[:len(frames) // 2]
        g_b = y_true_goh[len(frames) // 2:]
        nll_1 = log_loss(g_a, a)
        nll_2 = log_loss(g_b, b)
        nlls.append(nll_1 - nll_2)
    return observed_differences, np.mean(nlls >= observed_differences)


def stat_sig_brier(y_true, y_pred_1, y_pred_2, N_BOOTSTRAP=100):
    observed_differences = brier_multi(y_true, y_pred_1) - brier_multi(y_true, y_pred_2)
    nlls = list()
    for i in tqdm(range(N_BOOTSTRAP)):
        idx_data_1 = np.random.randint(len(y_pred_1), size=len(y_pred_1))
        idx_data_2 = np.random.randint(len(y_pred_2), size=len(y_pred_2))
        frames1 = y_pred_1[idx_data_1,:]
        y_true1 = y_true[idx_data_1, :]
        frames2 = y_pred_2[idx_data_2,:]
        y_true2 = y_true[idx_data_2, :]
        frames = np.concatenate([frames1, frames2], axis=0)
        y_true_goh = np.concatenate([y_true1, y_true2], axis=0)
        idx_data = np.random.randint(len(y_true_goh), size=len(y_true_goh))
        #
        frames = frames[idx_data]
        y_true_goh = y_true_goh[idx_data]
        a = frames[:len(frames)//2]
        b = frames[len(frames)//2:]
        g_a = y_true_goh[:len(frames)//2]
        g_b = y_true_goh[len(frames)//2:]
        nll_1 = brier_multi(g_a, a)
        nll_2 = brier_multi(g_b, b)
        nlls.append(nll_1 - nll_2)
    return observed_differences, np.mean(nlls >= observed_differences)


def stat_sig_ece(y_true, y_pred_1, y_pred_2, N_BOOTSTRAP=100):
    calib_erros = CalibrationErrors(y_true, y_pred_1, bin_size=1 / 20., min_samples=0)
    _, _, ece_pred1, _, _ = calib_erros.calculate_calibration_errors()
    calib_erros = CalibrationErrors(y_true, y_pred_2, bin_size=1 / 20., min_samples=0)
    _, _, ece_pred2, _, _ = calib_erros.calculate_calibration_errors()
    observed_differences = ece_pred1 - ece_pred2
    eces = list()
    for i in tqdm(range(N_BOOTSTRAP)):
        idx_data_1 = np.random.randint(len(y_pred_1), size=len(y_pred_1))
        idx_data_2 = np.random.randint(len(y_pred_2), size=len(y_pred_2))
        frames1 = y_pred_1[idx_data_1,:]
        y_true1 = y_true[idx_data_1, :]
        frames2 = y_pred_2[idx_data_2,:]
        y_true2 = y_true[idx_data_2, :]
        frames = np.concatenate([frames1, frames2], axis=0)
        y_true_goh = np.concatenate([y_true1, y_true2], axis=0)
        idx_data = np.random.randint(len(y_true_goh), size=len(y_true_goh))
        #
        frames = frames[idx_data]
        y_true_goh = y_true_goh[idx_data]
        a = frames[:len(frames)//2]
        b = frames[len(frames)//2:]
        g_a = y_true_goh[:len(frames)//2]
        g_b = y_true_goh[len(frames)//2:]
        calib_erros = CalibrationErrors(g_a, a, bin_size=1 / 20., min_samples=0)
        _, _, ece_1, _, _ = calib_erros.calculate_calibration_errors()
        calib_erros = CalibrationErrors(g_b, b, bin_size=1 / 20., min_samples=0)
        _, _, ece_2, _, _ = calib_erros.calculate_calibration_errors()
        eces.append(ece_1 - ece_2)
    return observed_differences, np.mean(eces >= observed_differences)

def stat_sig_top_k(y_true, y_pred_1, y_pred_2, N_BOOTSTRAP=100):
    observed_differences = top_k_accuracy(y_true, y_pred_1) - top_k_accuracy(y_true, y_pred_2)
    nlls = list()
    for i in tqdm(range(N_BOOTSTRAP)):
        idx_data_1 = np.random.randint(len(y_pred_1), size=len(y_pred_1))
        idx_data_2 = np.random.randint(len(y_pred_2), size=len(y_pred_2))
        frames1 = y_pred_1[idx_data_1,:]
        y_true1 = y_true[idx_data_1, :]
        frames2 = y_pred_2[idx_data_2,:]
        y_true2 = y_true[idx_data_2, :]
        frames = np.concatenate([frames1, frames2], axis=0)
        y_true_goh = np.concatenate([y_true1, y_true2], axis=0)
        idx_data = np.random.randint(len(y_true_goh), size=len(y_true_goh))
        #
        frames = frames[idx_data]
        y_true_goh = y_true_goh[idx_data]
        a = frames[:len(frames)//2]
        b = frames[len(frames)//2:]
        g_a = y_true_goh[:len(frames)//2]
        g_b = y_true_goh[len(frames)//2:]
        nll_1 = top_k_accuracy(g_a, a)
        nll_2 = top_k_accuracy(g_b, b)
        nlls.append(nll_1 - nll_2)
    return observed_differences, np.mean(nlls >= observed_differences)