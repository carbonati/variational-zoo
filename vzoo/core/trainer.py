import os
import time
import numpy as np
import tensorflow as tf
from vzoo.losses.disentangled_losses import compute_loss
from vzoo.eval.disentangled_metrics import compute_disentanglement_scores
from vzoo.eval.writer_utils import get_metrics, get_writers, display_logs
from vzoo.models import model_utils
from vzoo.vis.visualize import save_latent_traversal
from vzoo import config


class Trainer:
    """Base Trainer for unsupervised models composed of an inference
    and generative model

    Parameters
    ----------
    model : vzoo.models.vae.VAE
        Variational autoencoder with trainable variables to perform a
        training step.
    loss_fn : function, str
        Function to compute loss or string of a supported loss function
        from vzoo.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer to perform a training step with.
    train_dataset : tf.data.Dataset
        Training data used to evaluate gradients and compute `loss_fn`.
    output_dir : str
        Root directory to save model checkpoints, logs, and visualizations.
    val_dataset : tf.data.Dataset (default=None)
        Validation data.
    dis_dataset : DisentangledDataset (default=None)
        Disentangled dataset to compute various disentanglement scores.
        To compute a disentanglement score
    optimizer_disc : tf.keras.optimizers.Optimizer
        Optimizer used to train discriminator to perform a gradient step.
        Required when using FactorVAE.
    traversal_fns : list[function] (default=None)
        List of functions to perform latent traversals between epochs
        for plotting.
    representation_fn : function (default=None)
        Returns r(x) to compute disentanglement scores.
    seed : int (default=None)
        Random state.
    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 train_dataset,
                 output_dir,
                 val_dataset=None,
                 dis_dataset=None,
                 optimizer_disc=None,
                 traversal_fns=None,
                 representation_fn=None,
                 seed=None,
                 **kwargs):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.output_dir = output_dir
        self.val_dataset = val_dataset
        self.dis_dataset = dis_dataset
        self.optimizer_disc = optimizer_disc
        self.traversal_fns = traversal_fns
        self.representation_fn = representation_fn
        self.seed = seed

        self.train_writer = None
        self.val_writer = None
        self.dis_writer = None
        self.model_dir = None
        self.log_dir = None
        self.vis_dir = None
        self._random_state = None
        self._latent_indices = kwargs.get('latent_indices')
        self._base_groups = ['train', 'val', 'test', 'dis']
        self._global_step = 0

        self._params = kwargs
        self._set_args()

    def _set_summary_writer(self):
        """Sets the train, validation, and disentanglement summary writers."""
        self.train_writer, self.val_writer, self.dis_writer = get_writers(self.log_dir)

    def _check_metrics(self, params):
        for group in self._base_groups:
            metric_name = f'{group}_metrics'
            if group == 'dis':
                params[metric_name] = params.get(metric_name,
                                                 get_metrics(config.DISENTANGLED_SCORES))
            else:
                params[metric_name] = params.get(metric_name,
                                                 get_metrics(config.DISENTANGLED_LOSSES))

    def _prepare_output(self):
        """Checks for model, logging, and visualization directories."""
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.vis_dir = os.path.join(self.output_dir, 'vis')
        for root in [self.model_dir, self.log_dir, self.vis_dir]:
            if not os.path.exists(root):
                os.makedirs(root)

    def _set_args(self):
        self._prepare_output()
        self._set_summary_writer()

        self.loss_fn = model_utils.check_loss_fn(self.loss_fn)
        if self.seed is not None:
            self._random_state = np.random.RandomState(self.seed)
        if self._latent_indices is None:
            if hasattr(self.model, 'latent_dim'):
                self._latent_indices = np.arange(np.minimum(8, self.model.latent_dim))
        if self.representation_fn is None:
            self.representation_fn = self._representation_fn

    def _save_model(self, model, model_dir):
        """Saves mode to disk."""
        verbose = self._global_step == 1
        model_utils.save_model(model, model_dir, verbose=verbose)

    def _representation_fn(self, model):
        """Returns r(x) to compute disentanglement scores."""
        if hasattr(model, 'inference_model'):
            return lambda x: model.inference_model(x)[0]
        elif hasattr(model, 'encode'):
            return lambda x: model.encode(x)[0]
        else:
            return lambda x: model(x)

    def _reset_states(self, params):
        """Resets the state of each metric between epochs."""
        for group in self._base_groups:
            for metric in params.get(f'{group}_metrics', {}).values():
                metric.reset_states()

    def _generate_animations(self, epoch, params={}):
        """Generates a latent traversal for a subset of samples."""
        for traversal_fn in self.traversal_fns:
            filename = f'latent_{traversal_fn.__name__}_{epoch}.gif'
            filepath = os.path.join(self.vis_dir, filename)
            save_latent_traversal(filepath,
                                  self.model,
                                  dataset=self.dis_dataset,
                                  traversal_fn=traversal_fn,
                                  latent_indices=self._latent_indices,
                                  random_state=self._random_state,
                                  params=params)

    # https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-540071844
    @tf.function
    def _train_step(self, x, **kwargs):
        self.train_step(x, **kwargs)

    def train_step(self, x, params={}):
        """Peforms an optimization step for a batch of training samples.

        Parameters
        ----------
        x : tf.Tensor
            Batch of training samples.
        params : dict (default={})
            Additional parameters to pass to the loss function.
        """
        with tf.GradientTape(persistent=True) as tape:
            losses = compute_loss(model=self.model,
                                  loss_fn=self.loss_fn,
                                  x=x,
                                  writer=self.train_writer,
                                  params=params)

        gradients = tape.gradient(losses['loss'], self.model.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.vae.trainable_variables))

        if self.optimizer_disc is not None:
            gradients_disc = tape.gradient(losses['disc_loss'],
                                           self.model.discriminator.trainable_variables)
            self.optimizer_disc.apply_gradients(
                zip(gradients_disc, self.model.discriminator.trainable_variables)
            )
        del tape

    def train(self,
              epochs=1,
              verbose=1,
              params={}):
        """Trains the model for a fixed number of epochs.

        epochs : int (default=1)
            Number of epochs (iterations) to train a model.
        verbose : bool (default=1)
            Verbosity.
        params : dict (default={})
            Additional parameters to pass to the loss function, disentanglement
            scores, latent traversal plotting, etc.
        """
        params = dict(params, **self._params)
        self._check_metrics(params)
        dis_score_fns = params.get('dis_score_fns')

        for epoch in range(1, epochs+1):
            tf.summary.experimental.set_step(epoch)
            start_time = time.time()

            for x_train in self.train_dataset:
                self._train_step(x_train, params=params)

            if self.val_writer is not None:
                for x_val in self.val_dataset:
                    loss = compute_loss(self.model,
                                        self.loss_fn,
                                        x_val,
                                        writer=self.val_writer,
                                        training=False,
                                        params=params)

            if self.dis_dataset is not None and dis_score_fns is not None:
                if params.get('dis_metrics'):
                    rx = self.representation_fn(self.model) # r(x)
                    compute_disentanglement_scores(
                        writer=self.dis_writer,
                        dataset=self.dis_dataset,
                        model=rx,
                        score_fns=dis_score_fns,
                        params=params
                    )

                if self.traversal_fns is not None:
                    self._generate_animations(epoch, params=params)

            if verbose:
                display_logs(params['train_metrics'],
                             val_metrics=params.get('val_metrics'),
                             dis_metrics=params.get('dis_metrics'),
                             epochs=epochs,
                             elapsed_time=round(time.time() - start_time))
            self._reset_states(params)
            self._global_step += 1

            self._save_model(self.model, self.model_dir)
