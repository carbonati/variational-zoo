import numpy as np
from sklearn.preprocessing import StandardScaler
import vzoo.evaluation.metrics_utils as utils
from vzoo.evaluation import writer_utils

__all__ = [
    'compute_disentanglement_scores',
    'compute_bvae_score',
    'compute_dci_score',
    'compute_mig_score',
    'compute_mod_explicit_score',
    'compute_sap_score'
]


def compute_disentanglement_scores(writer,
                                   dataset,
                                   model,
                                   score_fns,
                                   num_samples=1000,
                                   batch_size=64,
                                   random_state=None,
                                   params=None):
    """Computes, and writes list of disentanglement scores.

    Parameters
    ----------
    writer : tf.summary.SummaryWriter
        Summary writer object for disentangled metrics.
    dataset : DisentangledDataset
        Dataset to generate and sample ground truth latent factors from.
    model : tf.keras.Model
        Encoder or some representation function r(x) that takes in input
        and returns the latent representation.
    score_fns: function, list, tuple
        List of functions to compute a disentanglement score.
        Each function should accept the following parameters:
            score_fn(dataset,
                     model,
                     num_samples,
                     batch_size,
                     random_state,
                     params)
    num_samples : int (default=1000)
        Number of samples to generate for evaluation. If a score requires
        train and test samples they will both be of size 'num_samples'.
    batch_size : int (default=64)
        Number of samples to generate per batch.
    random_state : np.random.RandomState
        Pseudo-random number generator.
    params : dict (default=None)
        A dictionary of vzoo and disentanglement score parameters.
        When passed in all scores will be updated in a tf.keras.metrics
        mapped to 'dis_metrics'.
    """
    params = params or {}
    scores_dict = {}
    if not isinstance(score_fns, (list, tuple, np.ndarray)):
        score_fns = [score_fns]

    for score_fn in score_fns:
        scores = score_fn(dataset,
                          model,
                          num_samples=num_samples,
                          batch_size=batch_size,
                          random_state=random_state,
                          params=params)
        scores_dict.update(scores)

    # create new list of metrics each function call
    params['dis_metrics'] = writer_utils.get_metrics(scores_dict.keys())
    writer_utils.log_metrics(writer, scores_dict, params['dis_metrics'])


def compute_bvae_score(dataset,
                       model,
                       num_samples=10000,
                       batch_size=64,
                       random_state=None,
                       params=None):
    """beta-VAE Disentanglement Metric

    Section 3 of "beta-VAE: Learning Basic Visual Concepts with a Constrained
    Variational Framework" (https://openreview.net/references/pdf?id=Sy2fzU9gl).

    Factor change classification
    ----------------------------
    Let L denote the number of latent variable,

        (1) Randomly sample ground truth factors v_li and v_lj such that a
            single factor v_k remains unchanged for both.
        (2) Generate observations x_li and x_lj from v_li and v_lj.
        (3) Generate z_li = mu(x_li) and z_lj = mu(x_lj), where mu(x) returns
            the mean latent vector from the representation function (encoder).
        (4) Calculate z_diff as,
                z_diff = 1/L * sum_l (|z_li - z_lj|).
        (5) bvae_score = ACC(y, p(y|z_diff))
                where p is a linear classifier trained to predict which
                factor y remained unchanged when generating z_diff.

    Parameters
    ----------
    dataset : DisentangledDataset
        Dataset to generate and sample ground truth latent factors.
    model : tf.keras.Model
        Encoder or representation function r(x) that takes in an input
        observation and returns the latent representation.
    num_samples : (default=10000)
        Number of latent representations to generate.
    batch_size : int (default=64)
        Number of samples to generate per batch.
    random_state : np.random.RandomState
        Pseudo-random number generator.
    params : dict (default=None)
        bvae_lr_params :
        scale :

    Returns
    -------
    scores_dict : dict
        bvae_score : beta-VAE disentanglement score.
    """
    scores_dict = {}

    Z_diff_train, y_train = utils.generate_factor_change(
        dataset,
        model,
        num_samples,
        batch_size=batch_size,
        random_state=random_state
    )
    Z_diff_test, y_test = utils.generate_factor_change(
        dataset,
        model,
        num_samples,
        batch_size=batch_size,
        random_state=random_state
    )

    bvae_score = utils.compute_factor_change_accuracy(
        Z_diff_train,
        y_train,
        Z_diff_test,
        y_test
    )
    scores_dict['bvae_score'] = bvae_score
    return scores_dict


def compute_dci_score(dataset,
                      model,
                      num_samples=1000,
                      batch_size=64,
                      random_state=None,
                      params=None):
    """Disentanglement, Completeness, and Informativeness (DCI)

    Section 2 of "A Framework for the Quantitative Evaluation of Disentangled
    Representations" (https://openreview.net/pdf?id=By-7dz-AZ).

    Let P be a (D, K) matrix where the each ij^th element denotes the
    probability of latent variable c_i of dimension D being important for
    predicting latent factor z_j of dim K. Let ro be a vector of length K
    with a weighted average from each factor.

    D = sum_i (ro_i * (1 - H(P_i)))
    C = sum_j (ro_j * (1 - H(P_j)))
    I = E(z_j, f_j(c)), i.e. the prediction error to predict z_j from c.

    Parameters
    ----------
    dataset : DisentangledDataset
        Dataset to generate and sample ground truth latent factors.
    model : tf.keras.Model
        Encoder or representation function r(x) that takes in an input
        observation and returns the latent representation.
    num_samples : (default=None)
        Number of latent representations to generate.
    batch_size : int (default=64)
        Number of samples to generate per batch.
    random_state : np.random.RandomState
        Pseudo-random number generator.

    Returns
    -------
    scores_dict : dict
        dci_info_score : Informativeness score.
        dci_comp_score : Completeness score.
        dci_dis_score : Disentanglement score.
    """
    scores_dict = {}
    params = params or {}

    x_train, y_train = utils.generate_factor_representations(
        dataset,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
        random_state=random_state
    )
    x_test, y_test = utils.generate_factor_representations(
        dataset,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
        random_state=random_state
    )
    train_error, test_error, P = utils.fit_info_clf(x_train,
                                              y_train,
                                              x_test,
                                              y_test,
                                              params=params)

    scores_dict['dci_info_score'] = test_error
    scores_dict['dci_comp_score'] = utils.compute_completeness(P)
    scores_dict['dci_dis_score'] = utils.compute_disentanglement(P)

    return scores_dict


def compute_mig_score(dataset,
                      model,
                      num_samples=1000,
                      batch_size=64,
                      workers=16,
                      random_state=None,
                      params=None):
    """Mutual Information Gap (MIG)

    Equation 6 (section 4.1) of "Isolating Sources of Disentanglement in
    Variational Autoencoders" (https://arxiv.org/pdf/1802.04942.pdf).

    Let z_j denote the jth latent variable, v_j the jth ground truth latent
    factor, and K the number of ground truth factors.

    MIG = 1/K * sum_k (1 / H(v_k)) * (argmax_jk( I(z_jk ; v_k) ) - argmax_j( I(z_j ; v_k)) )

    Where argmax_jk( I(z_jk ; v_k) ) denotes is the highest mutual
    information (MI) between the jk^th latent and k^th factor.
    argmax_j( I(z_j ; v_k) ) denotes the 2nd highest MI between the jth latent
    and kth factor.

    Parameters
    ----------
    dataset : DisentangledDataset
        Dataset to generate and sample ground truth latent factors.
    model : tf.keras.Model
        Encoder or representation function r(x) that takes in an input
        observation and returns the latent representation.
    num_samples : (default=None)
        Number of latent representations to generate.
    batch_size : int (default=64)
        Number of samples to generate per batch.
    random_state : np.random.RandomState
        Pseudo-random number generator.
    params : dict (default=None)
        bins : Discrete number of bins to encode each latent variable.

    Returns
    -------
    scores_dict : dict
        mig_score : Mutual Information Gap score.
    """
    params = params or {}
    bins = params.get('bins', 10)
    scores_dict = {}

    z, v = utils.generate_factor_representations(
        dataset,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
        random_state=random_state
    )
    z_binned = utils.descretize(z, bins=bins)

    H = utils.calculate_entropy(v)
    I = utils.calculate_mutual_info(z_binned, v)
    I_sorted = np.sort(I, axis=0)[::-1]

    scores_dict['mig_score'] = np.mean( ( I_sorted[0] - I_sorted[1] ) / H )
    return scores_dict


def compute_mod_explicit_score(dataset,
                               model,
                               num_samples,
                               batch_size=64,
                               random_state=None,
                               params=None):
    """Modularity and Explicitness scores.

    Equation 2 (Section 3) of "Learning Deep Disentangled Embeddings with
    the F-Statistic Loss" (https://arxiv.org/pdf/1802.05312.pdf).

    Let m_if denote the mutual information (MI) between the latent i and
    factor j. Then t is a zero matrix of the same size called a "template"
    where each row has only one non-zero element, the highest m_ij for latent i.
    Let N denotes the number of factors

    modularity = sum_i (1 - delta_i) / N
        s.t. delta_i = sum_f ((m_if - t_if)^2) / (theta^2_i * (N - 1))

    explicitness = sum_j (AUC(z_j, f_j(v))) / N
        where j is a factor index of z and k is an index on values of factor j.
        i.e. each factor is one hot encoded and we're trying to predict
        with each latent, then taking the mean AUC to evaluate performance.



    Parameters
    ----------
    dataset : DisentangledDataset
        Dataset to generate and sample ground truth latent factors.
    model : tf.keras.Model
        Encoder or representation function r(x) that takes in an input
        observation and returns the latent representation.
    num_samples : (default=None)
        Number of latent representations to generate.
    batch_size : int (default=64)
        Number of samples to generate per batch.
    random_state : np.random.RandomState
        Pseudo-random number generator.
    params : dict (default=None)
        bins : Discrete number of bins to encode each latent variable.

    Returns
    -------
    scores_dict : dict
        mod_score : Modularity score.
        eplicit_score : Explicitness score.
    """
    params = params or {}
    bins = params.get('bins', 10)
    scores_dict = {}

    x_train, y_train = utils.generate_factor_representations(
        dataset,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
        random_state=random_state
    )
    x_test, y_test = utils.generate_factor_representations(
        dataset,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
        random_state=random_state
    )

    x_train_binned = utils.descretize(x_train, bins=bins)
    MI = utils.calculate_mutual_info(x_train_binned, y_train)

    scl = StandardScaler()
    x_train = scl.fit_transform(x_train)
    x_test = scl.transform(x_test)

    explicit_score, val_explicit_score = utils.compute_explicitness(
        x_train,
        y_train,
        x_test,
        y_test,
        params=params
    )

    scores_dict["mod_score"] = utils.compute_modularity(MI)
    scores_dict['explicit_score'] = val_explicit_score

    return scores_dict


def compute_sap_score(dataset,
                      model,
                      num_samples,
                      batch_size=64,
                      random_state=None,
                      params=None):
    """Separated Attribute Predictability (SAP) score.

    Variational Inference of Disentangled Latent Concepts from Unlabelled
    Observations
    (https://arxiv.org/pdf/1711.00848.pdf)

    (i)  Constuct a d x k (latent x factor) score matrix s.t. the ij^th entry
         represents the amount variance in the factor j explained by the latent
         i indicated by their R^2 score. OR the classification accuracy of a
         SVM fit with the ith latent to predict the jth factor.
    (ii) For each factor the SAP score is calculated as the mean difference
         of the 2 most predictive latents.

    `num_samples` refered to as L in the paper.
    """
    params = params or {}
    classification = params.get('classification', False)
    scores_dict = {}

    x_train, y_train = utils.generate_factor_representations(
        dataset,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
        random_state=random_state
    )
    x_test, y_test = utils.generate_factor_representations(
        dataset,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
        random_state=random_state
    )

    score_matrix = utils.compute_score_matrix(x_train,
                                              y_train,
                                              x_test,
                                              y_test,
                                              classification=classification)
    scores_dict["sap_score"] = utils.compute_top_mean_diff(score_matrix)
    return scores_dict

