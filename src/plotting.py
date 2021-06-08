import os

import matplotlib.pyplot as plt


def plot_loss_curves(history, output_path=None, y_lim=(0, 150)):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(y_lim)
    plt.legend(['train', 'validation'], loc='upper left')

    if output_path is not None:
        plt.savefig(os.path.join(output_path, 'loss_curves.png'), format='png', dpi=300)
    plt.show()


def plot_rul(expected, predicted):
    plt.figure()
    plt.plot(range(len(expected)), expected, label='Expected')
    plt.plot(range(len(predicted)), predicted, label='Predicted')
    plt.legend()
    plt.xlabel("Time (num samples)")
    plt.ylabel("RUL")


def plot_rul_confidence_interval(df):
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


def plot_signal_filtering(df, signal_name, alpha=0.05, w=10):
    signal = df[signal_name][:200]
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
