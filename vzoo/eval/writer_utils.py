# tensorboard utils
import os
import datetime
import tensorflow as tf
from vzoo import config


def get_writers(log_dir):
    """
    Returns a train, validation, and disentanglement scoring tf.summary.writer
    for logging in tensorboard.
    """
    train_log_dir = os.path.join(log_dir, 'train')
    val_log_dir = os.path.join(log_dir, 'val')
    dis_log_dir = os.path.join(log_dir, 'dis')

    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)
    dis_writer = tf.summary.create_file_writer(dis_log_dir)

    return train_writer, val_writer, dis_writer


def get_metrics(metrics=None):
    """Returns a dict of names, tf.keras.metric.Mean pairs to log."""
    metric_dict = {}
    if metrics is None:
        metrics = config.DISENTANGLED_LOSSES
    metrics = [metrics] if isinstance(metrics, str) else metrics
    for name in metrics:
        metric_dict[name] = tf.keras.metrics.Mean(name, dtype=tf.float32)
    return metric_dict


def _metric_str_list(metric_dict):
    """Returns a list of formated strings given a dict of metrics."""
    metric_dict = metric_dict or {}
    metric_str_list = []
    for k, v in metric_dict.items():
        if v.count > 0:
            metric_str_list.append(f'{k}: {v.result():.4f}')
    return metric_str_list


def log_metrics(writer, scores_dict, metric_dict=None):
    """Save scores to a summary writer and tf.keras.metrics instance."""
    metric_dict = metric_dict or {}
    with writer.as_default():
        for k, v in scores_dict.items():
            tf.summary.scalar(k, v)
            metric = metric_dict.get(k)
            if metric:
                metric.update_state(v)


def display_logs(train_metrics,
                 val_metrics=None,
                 dis_metrics=None,
                 epochs=None,
                 elapsed_time=0):
    """Displays the train, validation, and disentanglement metrics logged during
    training.
    """
    train_metrics = _metric_str_list(train_metrics)
    val_metrics = _metric_str_list(val_metrics)
    dis_metrics = _metric_str_list(dis_metrics)

    template = 'Epoch {epoch}/{epochs} - {s}s - {train_metrics}'
    if len(val_metrics) > 0:
        template += ' - val_{val_metrics}'
    if len(dis_metrics):
        template += ' - {dis_metrics}'
    template += '\n'

    print(template.format(
        epoch=tf.summary.experimental.get_step(),
        epochs=epochs,
        s=elapsed_time,
        train_metrics=', '.join(train_metrics),
        val_metrics=', val_'.join(val_metrics),
        dis_metrics=', '.join(dis_metrics)
    ))
