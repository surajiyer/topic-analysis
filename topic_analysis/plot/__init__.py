from adjustText import adjust_text
from matplotlib.dates import DateFormatter, WeekdayLocator
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram


def plot_phrases_over_time(
    df, *, text_column, date_column, id_column,
    phrases, time_interval='daily', ma_window_size=2,
    lowercase=True, sig_val=1.96,
    annotate_abs_delta_threshold=10, **plot_kws):
    """
    Plot the trend lines of phrase usage over time.

    Parameters
    ----------
    df : pandas.DataFrame
        text dataset

    text_column : str
        name of the text column in df

    date_column : str
        name of the date / datetime column in df

    id_column : str
        name of the document id column in df

    phrases : Iterable[Union[str, Iterable[str]]
        which phrases (str) to plot and / or which
        topics (Iterable[str]) to plot? For topics,
        the average trend line over the topic keywords
        is calculated. Note: phrases are case-sensitive
        unless `lowercase=True` which is the default
        behavior.

    time_interval : str, default='daily'
        Time interval over which to plot trends.
        Possible values: ('hourly', 'daily',
        'weekly', 'monthly', 'quarterly', 'yearly')

    ma_window_size : int, default=2
        Change percent of phrase count is calculated w.r.t 
        a moving average over last `ma_window_size` intervals
        of time.

    lowercase : bool, default=True
        Lowercase the text before searching for the phrases.

    sig_val : float, default=1.96
        Significance value. Any phrase / topic with z-score > sig_val
        or < -sig_val will be colored else will be greyed out in the
        plot. The default 1.96 means 95% confidence interval on z-score.
        You can also use 2.33 for 99%.

    annotate_abs_delta_threshold : int, default=10
        Threshold on actual delta to annotate in plot.

    plot_kws : kwargs
        additional arguments for pandas.DataFrame.plot()
    """
    # time interval
    TIME_INTERVALS = ('hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly')
    time_interval = time_interval if time_interval\
        in TIME_INTERVALS else 'daily'
    time_interval = pd.to_datetime(pd.Series(df[date_column]))\
        .dt.to_period(time_interval[0].upper())

    # calculate keyword trends
    phrases_trends_df = []
    text_column = df[text_column].str.lower()\
        if lowercase else df[text_column]

    i = 0
    for phrase in phrases:
        if isinstance(phrase, str):
            mask = rf'\b{phrase}\b'
        else:
            # assume phrase is iterable
            try:
                mask = '|'.join([rf'\b{w}\b' for w in phrase])
                phrase = f'Topic #{i}'
                i += 1
            except TypeError:
                raise ValueError('phrases contains non-string and / or non-iterable objects')

        _ = df[text_column.str.contains(mask)]\
            .groupby(time_interval)\
            .agg(count=pd.NamedAgg(column=id_column, aggfunc="count"))\
            .reset_index()\
            .assign(phrase=phrase)

        # calculate moving average frequency of phrase(s)
        # in previous time window of size `ma_window_size`
        _[['moving_avg', 'moving_std']] = _['count']\
            .shift(1)\
            .fillna(0.)\
            .rolling(window=ma_window_size, min_periods=1)\
            .agg(['mean', 'std'])

        # add result to list
        phrases_trends_df.append(_)

    # concatenate list of results to single dataframe
    phrases_trends_df = pd.concat(phrases_trends_df)

    # calculate actual delta
    phrases_trends_df['actual_delta'] =\
        phrases_trends_df['count'] - phrases_trends_df['moving_avg']

    # calculate percentage of change of frequency in
    # current time period compared to previous time
    # window moving average
    phrases_trends_df['change %'] =\
        phrases_trends_df['actual_delta'] * 100. / phrases_trends_df['moving_avg']

    # calculate z-score, i.e., standardized phrase usage rates
    phrases_trends_df['z-score'] =\
        (phrases_trends_df['actual_delta'] / phrases_trends_df['moving_std'])\
        .replace([-np.inf, np.inf], np.nan)

    # check if topic trend is outlier at the current time period
    phrases_trends_df['outlier'] = phrases_trends_df['z-score']\
        .apply(lambda z: z < -sig_val or z > sig_val)
#     display(phrases_trends_df)

    # plots
    fig, ax = plt.subplots(3, 1)

    # set color map
    if 'cmap' not in plot_kws:
        cmap = cm.get_cmap('tab20')
        plot_kws['cmap'] = cmap

    # set legend to false for individual axes
    # we want only one legend for the full figure
    plot_kws['legend'] = False

    if 'ax' in plot_kws:
        plot_kws.pop('ax')

    # set figure title
    fig.suptitle(plot_kws.pop('title', 'Phrase Usage Rate Analysis'), fontsize=16)

    # plot trend line
    ax[0] = phrases_trends_df\
        .pivot(index=date_column, columns='phrase', values=['count'])\
        .plot(ax=ax[0], **plot_kws)
#     ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].grid(True)
    ax[0].set_title("Usage rate")

    # add one common legend for complete figure
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, [_[8:-1] for _ in labels], loc='center left', bbox_to_anchor=(1, 0.5))

    # plot change percentage
    ax[1] = phrases_trends_df\
        .pivot(index=date_column, columns='phrase', values=['change %'])\
        .plot(ax=ax[1], cmap=plot_kws['cmap'], legend=plot_kws['legend'])
#     ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].grid(True)
    ax[1].set_title("Usage rate change percentage (%)")

    # annotation of actual difference in change % graph
    top_n_change = phrases_trends_df\
        .assign(abs_change_perc=phrases_trends_df['change %'].apply(abs))\
        .replace([np.inf, -np.inf], np.nan)\
        .dropna()\
        .sort_values('abs_change_perc', ascending=False).iloc[:30]
    top_n_change = top_n_change[[date_column, 'change %', 'actual_delta']]
    texts = []
    for x, y, z in top_n_change.itertuples(index=False, name=None):
        if abs(z) >= annotate_abs_delta_threshold:
    #         ax[1].annotate(f'+{z}' if z >= 0. else f'{z}', xy=(x, y), xytext=(x, y + 2))
            texts += [ax[1].text(x, y, f'+{z}' if z >= 0. else f'{z}')]
    adjust_text(texts, ax=ax[1], only_move={'points':'y', 'texts':'y'},
                expand_points=(1.8, 1.8),
                arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    ax[1].set_xlabel("Annotations are actual volume delta between current time period and moving average.")

    # plot z-score bubble plot
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [handles[i].get_color() for i in range(len(handles))]
    texts = []
    for i, (phrase, df_g) in enumerate(phrases_trends_df.groupby(['phrase'])):
        df_g[date_column] = df_g[date_column].dt.to_timestamp()
        df_g = df_g[df_g['z-score'].notnull()]
#         display(df_g)

        # colored bubbles if outlier
        ax[2] = df_g[df_g['outlier']]\
            .plot.scatter(x=date_column, y='z-score',
#                           color=colors[i % len(colors)],
                          color=colors[i],
                          s=df_g.loc[df_g['outlier'], 'count'] * 4,
                          ax=ax[2], label=phrase,
                          legend=plot_kws['legend'], alpha=.5)

        # get texts for annotation
        for x, y, z in df_g.loc[df_g['outlier'], [date_column, 'z-score', 'actual_delta']]\
            .itertuples(index=False, name=None):
            if abs(z) >= annotate_abs_delta_threshold:
                texts += [ax[2].text(x, y, f'+{z}' if z >= 0. else f'{z}')]

        # greyed out bubbles if not outlier
        ax[2] = df_g[~df_g['outlier']]\
            .plot.scatter(x=date_column, y='z-score',
                          c='grey', s=df_g.loc[~df_g['outlier'], 'count'] * 2,
                          ax=ax[2], legend=False, alpha=.5)

    # annotate actual delta for outliers
    adjust_text(texts, ax=ax[2], expand_points=(1.8, 1.8),
                arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    ax[2].axhspan(-sig_val, sig_val, fill=False, linestyle='dashed')
#     lgnd = ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     # change the marker size manually for both lines
#     for handle in lgnd.legendHandles:
#         handle._sizes = [20]
    ax[2].tick_params(which='major', length=15)
    ax[2].xaxis.set_minor_locator(WeekdayLocator(interval=1))
    ax[2].xaxis.set_minor_formatter(DateFormatter('%W'))
    ax[2].grid(True)
    ax[2].set_title("Usage rate Z-score")
    ax[2].set_xlabel("Annotations are actual volume delta between current time period and moving average.")

    fig.tight_layout()
    return fig, ax


def plot_dendrogram(model, **kwargs):
    """
    Create linkage matrix and then plot the dendrogram.

    Parameters
    ----------
    model : sklearn.cluster.AgglomerativeClustering
        Hierarchical clustering scikit-learn model object

    kwargs : dict
        For more information: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
#     print(linkage_matrix)

    # Plot the corresponding dendrogram
    fig, ax = plt.subplots(1, 1, figsize=kwargs.pop('figsize', None))
    ax.set_title('Hierarchical Clustering Dendrogram')
    dendrogram(linkage_matrix, ax=ax, **kwargs)
    ax.set_xlabel("Number of points in node (or index of point if no parenthesis).")
    return fig, ax
