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
    plt.show()


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
