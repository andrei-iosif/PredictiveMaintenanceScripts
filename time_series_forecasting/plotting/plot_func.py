import matplotlib.pyplot as plt


def time_series_plot(series):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(len(series)), series)
    ax2.hist(series)
    plt.show()


def plot_forecasting(expected, predictions_list, legend=(), linestyles=(), title="", output_path=None):
    """
    Plot the forecast results of multiple models.
    :param expected: the actual time series values
    :param predictions_list: prediction results from multiple models
    :param legend: labels
    """

    marker_1 = 'o' if len(expected) < 15 else ''
    marker_2 = '*' if len(expected) < 15 else ''

    plt.plot(expected, label='Expected', marker=marker_1)
    for i in range(len(predictions_list)):
        plt.plot(predictions_list[i], label=legend[i], linestyle=linestyles[i], marker=marker_2)
    plt.legend()

    plt.xlabel("Time (num observations)")
    plt.ylabel("Value")
    plt.title(title)

    if output_path is not None:
        plt.savefig(output_path, format='png', dpi=300)

    plt.show()
