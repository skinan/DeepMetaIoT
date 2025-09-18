import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow import device
import tensorflow as tf

# Hide GPU from visible devices
# tf.config.set_visible_devices([], "GPU")
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import (
    Dense,
    Input,
    Conv1D,
    GlobalAveragePooling1D,
    MaxPooling1D,
    BatchNormalization,
    LayerNormalization,
    Bidirectional,
    GRU,
    concatenate,
    Dropout,
    Add,
    ReLU,
)


from sentence_transformers import SentenceTransformer


def conv_block(input_layer, filter_size):
    activation = "relu"
    padding = "causal"
    conv = Conv1D(128, filter_size, activation=activation, padding=padding)(input_layer)
    conv = Conv1D(128, filter_size, activation=activation, padding=padding)(conv)
    conv = Conv1D(128, filter_size, activation=activation, padding=padding)(conv)
    conv = MaxPooling1D(data_format="channels_last")(conv)
    conv = Conv1D(64, filter_size, activation=activation, padding=padding)(conv)
    conv = Conv1D(64, filter_size, activation=activation, padding=padding)(conv)
    conv = Conv1D(64, filter_size, activation=activation, padding=padding)(conv)
    conv = MaxPooling1D(data_format="channels_last")(conv)
    conv = Conv1D(64, filter_size, activation=activation, padding=padding)(conv)
    conv = Conv1D(64, filter_size, activation=activation, padding=padding)(conv)
    conv = Conv1D(64, filter_size, activation=activation, padding=padding)(conv)
    conv = GlobalAveragePooling1D(data_format="channels_last")(conv)
    return conv


def time_series_encoder(ts_input_layer):
    lstm = Bidirectional(
        GRU(
            128,
            return_sequences=True,
        )
    )(ts_input_layer)
    lstm = BatchNormalization()(lstm)
    lstm = Bidirectional(
        GRU(
            128,
            return_sequences=True,
        )
    )(lstm)
    lstm = BatchNormalization()(lstm)
    lstm = Bidirectional(GRU(64, return_sequences=False))(lstm)
    lstm = BatchNormalization()(lstm)

    l11 = conv_block(input_layer=ts_input_layer, filter_size=11)
    l7 = conv_block(input_layer=ts_input_layer, filter_size=7)
    l5 = conv_block(input_layer=ts_input_layer, filter_size=5)
    l3 = conv_block(input_layer=ts_input_layer, filter_size=3)

    conv_concat = concatenate([l11, l7, l5, l3])  # Depth Concatenate

    merge = concatenate([lstm, conv_concat])
    return merge


def cross_residual_dense_block(x, cross_x):
    Fx = Dense(units=x.shape[-1], activation=None)(x)  # WEIGHT LAYER
    Fx = Dropout(0.3)(Fx)
    Fx = LayerNormalization()(Fx)  # NORMALIZATION
    Fx = ReLU()(Fx)
    Fx = Dense(units=x.shape[-1] // 2, activation=None)(Fx)  # WEIGHT LAYER
    Fx = LayerNormalization()(Fx)  # NORMALIZATION
    Fx = ReLU()(Fx)
    Fx = Dense(units=cross_x.shape[-1], activation=None)(Fx)
    Fx = LayerNormalization()(Fx)
    out = Add()([Fx, cross_x])
    out = Dropout(0.3)(out)
    out = LayerNormalization()(out)
    out = ReLU()(out)
    return out


def evaluate_model(model_name, history):
    val_accuracy = [x * 100 for x in history.history["val_accuracy"]]
    val_f1_macro = [x * 100 for x in history.history["val_f1_score_macro"]]
    val_f1_weighted = [x * 100 for x in history.history["val_f1_score_avg"]]
    top_epochs = 5
    # Sort the validation accuracies to get the best 5 epochs
    best_val_accuracies = sorted(val_accuracy, reverse=True)[:top_epochs]
    best_f1_macro = sorted(val_f1_macro, reverse=True)[:top_epochs]
    best_f1_weighted = sorted(val_f1_weighted, reverse=True)[:top_epochs]
    print(
        "Best Accuracy: ",
        max(best_val_accuracies),
        "\n",
        "Best F1-Macro: ",
        max(best_f1_macro),
    )
    results = []
    for accuracy, f1_macro, f1_weighted in zip(
        best_val_accuracies, best_f1_macro, best_f1_weighted
    ):
        results.append(
            {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate mean and standard deviation of the top 5 epochs
    mean_accuracy = results_df["accuracy"].mean()
    std_accuracy = results_df["accuracy"].std()
    mean_f1_macro = results_df["f1_macro"].mean()
    std_f1_macro = results_df["f1_macro"].std()
    mean_f1_weighted = results_df["f1_weighted"].mean()
    std_f1_weighted = results_df["f1_weighted"].std()
    results_list = [
        model_name,
        mean_accuracy,
        std_accuracy,
        mean_f1_macro,
        std_f1_macro,
        mean_f1_weighted,
        std_f1_weighted,
        max(best_val_accuracies),
        max(best_f1_macro),
    ]
    import csv

    # Make sure the parent directory exists
    results_file_name = f"../results/TS-M-results.csv"
    os.makedirs(os.path.dirname(results_file_name), exist_ok=True)

    with open(results_file_name, "a", newline="") as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(results_list)


def main(dataframe, train_task_name):
    X = dataframe.drop(
        labels=[
            "database_id",
            "metadata_0",
            "metadata_1",
            "metadata_2",
            "metadata_3",
            "metadata_4",
            "metadata_5",
            "metadata_6",
            "label",
        ],
        axis=1,
    )
    X_meta = pd.DataFrame({"metadata": []})
    X_meta["metadata"] = dataframe["metadata_0"].astype(str)

    y = dataframe["label"]
    X_train, X_test, y_train, y_test, X_meta_train, X_meta_test = train_test_split(
        X, y, X_meta, test_size=0.3, stratify=y, random_state=100
    )

    X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
    # Pre-train Embedding for Textual Meta-data
    pre_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    X_meta_train_embed = np.array(
        list(np.vstack(X_meta_train["metadata"].apply(pre_transformer.encode)))
    )
    X_meta_test_embed = np.array(
        list(np.vstack(X_meta_test["metadata"].apply(pre_transformer.encode)))
    )

    # Define the model

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    text_input_layer = Input(shape=X_meta_train_embed.shape[1:])
    ts_input_layer = Input(shape=X_train.shape[1:])

    ts_encoder = time_series_encoder(ts_input_layer)

    x1 = ts_encoder
    x2 = text_input_layer

    for _ in range(1):
        temp = cross_residual_dense_block(x1, x2)
        x2 = cross_residual_dense_block(x2, x1)
        x1 = temp

    concat_fusion = concatenate([x1, x2])

    output_layer = Dense(units=len(np.unique(y_train)), activation="softmax")(
        concat_fusion
    )
    optimizer = Adam(learning_rate=0.01)

    loss = CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.0,
    )

    model = Model(
        inputs=[ts_input_layer, text_input_layer],
        outputs=output_layer,
    )

    f1_score_macro = tf.keras.metrics.F1Score(average="macro", name="f1_score_macro")
    f1_score_avg = tf.keras.metrics.F1Score(average="weighted", name="f1_score_avg")

    metrics = [
        "accuracy",
        f1_score_macro,
        f1_score_avg,
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(model.summary())
    history = model.fit(
        [X_train, X_meta_train_embed],
        y_train_encoded,
        epochs=200,
        validation_data=(
            [X_test, X_meta_test_embed],
            y_test_encoded,
        ),
        # callbacks=callbacks,
    )
    # model.save("DeepMetaIoT-Meta")  # Save the model
    evaluate_model(train_task_name, history=history)


if __name__ == "__main__":
    for _ in range(1):
        with device("/GPU:0"):
            dataframe = pd.read_csv("datasets/TS-M.csv")
            main(dataframe, "DeepMetaIoT TS-M")
