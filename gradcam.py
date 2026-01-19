import tensorflow as tf
import numpy as np

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            # Conv layers have 4D output: (batch, H, W, C)
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No convolutional layer found in model")


def grad_cam_densenet(model, img_array, class_idx):
    last_conv_layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        model.inputs,
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # 🔥 Handle multi-output models
        if isinstance(predictions, list):
            predictions = predictions[0]

        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def detect_orientation(heatmap):
    h, w = heatmap.shape
    scores = {
        "Left": heatmap[:, :w // 2].sum(),
        "Right": heatmap[:, w // 2:].sum(),
        "Top": heatmap[:h // 2, :].sum(),
        "Bottom": heatmap[h // 2:, :].sum()
    }
    return max(scores, key=scores.get)

