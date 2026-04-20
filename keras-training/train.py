import os, gc
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

# ==========================================
# 1. CONFIG
# ==========================================
TRAIN_PATH  = os.environ.get("PATH_TRAIN_OUT", "/data/processed/train")
MODEL_PATH  = os.environ.get("MODEL_PATH",     "/data/models")
IMG_SIZE    = 150
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", "16"))
EPOCHS      = int(os.environ.get("EPOCHS",     "10"))
CLASS_NAMES = ['Mountain', 'Glacier', 'Street', 'Sea', 'Forest', 'Buildings']

os.makedirs(MODEL_PATH, exist_ok=True)

def find_parquet(folder):
    for f in sorted(os.listdir(folder)):
        if f.endswith(".parquet"):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"Aucun .parquet dans {folder}")

PARQUET_FILE = find_parquet(TRAIN_PATH)

# ==========================================
# 2. SPLIT
# ==========================================
print("Lecture des labels...", flush=True)
pf     = pq.ParquetFile(PARQUET_FILE)
labels = pf.read(columns=["label"]).to_pandas()["label"].values.astype(np.int32)
n      = len(labels)

train_idx, test_idx = train_test_split(
    np.arange(n), test_size=0.2, random_state=42, stratify=labels
)

num_classes = len(np.unique(labels))
print(f"Train: {len(train_idx)} | Test: {len(test_idx)} | Classes: {num_classes}", flush=True)

# ==========================================
# 3. DATASET
# ==========================================
ROW_GROUP_SIZE = pf.metadata.row_group(0).num_rows
print(f"Row-groups: {pf.metadata.num_row_groups} x ~{ROW_GROUP_SIZE} lignes", flush=True)

def make_tf_dataset(indices, shuffle=False):
    sorted_idx = np.sort(indices)

    def row_generator():
        meta       = pf.metadata
        n_rg       = meta.num_row_groups
        boundaries = np.cumsum([0] + [meta.row_group(i).num_rows for i in range(n_rg)])

        idx_ptr = 0
        total   = len(sorted_idx)

        for rg in range(n_rg):
            rg_start = boundaries[rg]
            rg_end   = boundaries[rg + 1]

            lo = idx_ptr
            while idx_ptr < total and sorted_idx[idx_ptr] < rg_end:
                idx_ptr += 1
            rg_indices = sorted_idx[lo:idx_ptr]

            if len(rg_indices) == 0:
                continue

            table = pf.read_row_groups([rg], columns=["features", "label"])
            df    = table.to_pandas()
            del table

            for gidx in rg_indices:
                local = int(gidx - rg_start)
                row   = df.iloc[local]

                feat  = np.asarray(row["features"], dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE, 3)
                label = np.int32(row["label"])

                yield feat, label

            del df
            gc.collect()

    ds = tf.data.Dataset.from_generator(
        row_generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(),                      dtype=tf.int32),
        )
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=512, reshuffle_each_iteration=True)

    return ds.batch(BATCH_SIZE).prefetch(2)

print("Construction des datasets...", flush=True)
train_ds = make_tf_dataset(train_idx, shuffle=True)
test_ds  = make_tf_dataset(test_idx,  shuffle=False)

# ==========================================
# 4. METRIQUES
# ==========================================
def plot_metrics(history, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{name} - Precision')
    ax1.legend()

    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{name} - Perte')
    ax2.legend()

    out = os.path.join(MODEL_PATH, f"{name}_metrics.png")
    plt.savefig(out, dpi=80, bbox_inches='tight')
    plt.close(fig)

    print(f"Courbes: {out}", flush=True)


def plot_confusion(model, name):
    print("Calcul matrice de confusion...", flush=True)

    # Prédictions
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    y_true = []
    for _, y in test_ds:
        y_true.extend(y.numpy())
    y_true = np.array(y_true)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )

    ax.set_title(f'Confusion : {name}')
    ax.set_ylabel('Vrai')
    ax.set_xlabel('Predit')

    out = os.path.join(MODEL_PATH, f"{name}_confusion.png")
    plt.savefig(out, dpi=80, bbox_inches='tight')
    plt.close(fig)

    print(f"Confusion: {out}", flush=True)

# ==========================================
# 5. MODELE
# ==========================================
def build_cnn():
    return models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

# ==========================================
# 6. TRAIN
# ==========================================
print("\n" + "="*40 + "\nDEBUT : CNN\n" + "="*40, flush=True)

model = build_cnn()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    callbacks=[
        callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ],
    verbose=1,
)

plot_metrics(history, "CNN")
plot_confusion(model, "CNN")

save_path = os.path.join(MODEL_PATH, "intel_model_cnn.h5")
model.save(save_path)

print(f"Modele sauvegarde: {save_path}", flush=True)
print("Entrainement termine!", flush=True)

time.sleep(60)