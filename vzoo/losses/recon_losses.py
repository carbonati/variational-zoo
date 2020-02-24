import numpy as np
import tensorflow as tf

__all__ = [
    'compute_loss',
    'bernoulli_loss',
    'l2_loss',
]


def compute_loss(model,
                 loss_fn,
                 x,
                 writer=None,
                 training=True,
                 params={}):
    """Computes and records loss for a batch."""
    losses = loss_fn(model, x, params)

    if writer is not None:
        if training:
            metric_dict = params.get('train_metrics')
        else:
            metric_dict = params.get('val_metrics')
        log_metrics(writer, losses, metric_dict)

    return losses


def bernoulli_loss(x, y, from_logits=False):
    """Computes the bernoulli loss (log loss) between the targets (input)
    and predictions (reconstruction).
    """
    dim = np.prod(x.shape.dims[1:])
    x = tf.reshape(x, (-1, dim))
    y = tf.reshape(y, (-1, dim))

    if from_logits:
        loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=x, logits=y
            ),
            axis=1,
        )
    else:
        loss = - tf.reduce_sum(
            x * tf.math.log(y) + (1 - x) * tf.math.log(1 - y),
            axis=1
        )
    return tf.reduce_mean(loss)


def l2_loss(x, y, from_logits=False):
    """
    Calculates mean squared error (MSE) between the targets (input) and
    predictions (reconstructions).

    Most papers implemented in this repo use a bernoulli decoder, which works
    effectively for datasets like dSprites that have binary inputs for each
    pixel, however, for any dataset with a continuous input MSE almost always
    out performs bernoulli loss as shown in the experiments that I haven't
    done yet...
    """
    dim = np.prod(x.shape[1:])
    x = tf.reshape(x, (-1, dim))
    y = tf.reshape(y, (-1, dim))

    if from_logits:
        y = tf.nn.sigmoid(y)
    # sum across the squared errors of each column
    # average across the rows (samples)
    return tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis=1))

