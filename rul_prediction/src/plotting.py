import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(history, output_path=None, y_lim=(0, 150)):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(y_lim)
    plt.legend(['train', 'validation'], loc='upper left')

    if output_path is not None:
        plt.savefig(output_path, format='png', dpi=300)
    plt.show()


def plot_rul(expected, predicted, output_path=None):
    plt.figure()
    plt.plot(range(len(expected)), expected, label='Expected')
    plt.plot(range(len(predicted)), predicted, label='Predicted')
    plt.legend()
    plt.xlabel("Time (num samples)")
    plt.ylabel("RUL")

    if output_path is not None:
        plt.savefig(output_path, format='png', dpi=300)


def plot_rul_confidence_interval(df, output_path=None):
    x_1, y_1 = range(len(df['actual_RUL'])), df['actual_RUL']
    x_2, y_2 = range(len(df['predicted_RUL_mean'])), df['predicted_RUL_mean']
    cnf = df['predicted_RUL_std']

    fig, ax = plt.subplots()
    ax.plot(x_1, y_1, label='Expected')
    ax.plot(x_2, y_2, label='Predicted')

    ax.fill_between(x_2, (y_2 - cnf), (y_2 + cnf), color='orange', alpha=0.5)
    ax.legend()
    plt.xlabel("Time (cycles)")
    plt.ylabel("RUL")

    if output_path is not None:
        plt.savefig(output_path, format='png', dpi=300)


def plot_prediction_error(df, units=(11, 14, 15), err_line=10, output_path=None):
    fig, ax = plt.subplots()

    for unit in units:
        df_unit = df[df['unit'] == unit]
        df_unit_mean = df_unit.groupby(['cycle']).mean()

        x = range(len(df_unit_mean['err_mean']))
        y = df_unit_mean['err_mean']
        cnf = df_unit_mean['err_std']

        ax.plot(x, y, label=f'Unit {unit}')
        ax.fill_between(x, (y - cnf), (y + cnf), alpha=0.5)

    x = range(80)
    ax.plot(x, err_line * np.ones_like(x), color='red', linestyle='dashed')
    ax.plot(x, -err_line * np.ones_like(x), color='red', linestyle='dashed')
    ax.plot(x, np.zeros_like(x), color='red')

    ax.legend()
    plt.xlabel("Time (cycles)")
    plt.ylabel("RUL error (cycles)")

    if output_path is not None:
        plt.savefig(output_path, format='png', dpi=300)


def plot_signal_filtering(df, signal_name, signal_len=200, alpha=0.05, w=10, output_path=None):
    signal = df[signal_name][:signal_len]
    smooth_signal_es = signal.ewm(alpha=alpha, adjust=False).mean()
    smooth_signal_ma = signal.rolling(w, min_periods=1).mean()

    plt.figure()
    plt.title(f"Value of signal {signal_name}")
    plt.xlabel("Time (num observations)")
    plt.ylabel("Value")
    plt.plot(signal, label='Original')
    plt.plot(smooth_signal_es, label='Exponential smoothing')
    plt.plot(smooth_signal_ma, label='Moving average')
    plt.legend()

    if output_path is not None:
        plt.savefig(output_path, format='png', dpi=300)