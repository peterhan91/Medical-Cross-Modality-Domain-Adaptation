import tensorflow as tf

def fgm(model, x, eps=0.01, epochs=1, sign=True, clip_min=0., clip_max=1.):
    """
    Fast gradient method.
    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
    for details.  This implements the revised version since the original FGM
    has label leaking problem (https://arxiv.org/abs/1611.01236).
    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :return: A tensor, contains adversarial samples for each input.
    """
    xadv = tf.identity(x)

    ybar = model(xadv)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1) # Returns the index with the largest value across axes of a tensor.
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar, logits = model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,
                            name='fast_gradient')
    return xadv

def fgsm(net, x, eps):
    r"""Caffe implementation of the Fast Gradient Sign Method.
    This attack was proposed in
    net: The Caffe network. Must have its weights initialised already
         Makes the following assumptions
            - force_backward is set to "true" so that gradients are computed
            - Has two inputs: "data" and "label"
            - Has two outputs: "output" and "loss"
    x: The input data. We will find an adversarial example using this.
            - Assume that x.shape = net.blobs['data'].shape
    eps: l_{\infty} norm of the perturbation that will be generated
    Returns the adversarial example, as well as just the pertubation
        (adversarial example - original input)
    """

    shape_label = net.blobs['label'].data.shape
    dummy_label = np.zeros(shape_label)

    net.blobs['data'].data[0,:,:,:] = np.squeeze(x)
    net.blobs['label'].data[...] = dummy_label

    net.forward()
    net_prediction = net.blobs['output'].data[0].argmax(axis=0).astype(np.uint32)
    net.blobs['label'].data[...] = net_prediction

    data_diff = net.backward(diffs=['data'])
    grad_data = data_diff['data']
    signed_grad = np.sign(grad_data) * eps

    adversarial_x = x + signed_grad
    return adversarial_x, signed_grad