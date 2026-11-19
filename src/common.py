# common.py: الإعدادات المشتركة وتحميل البيانات وشبكة CNN أحادية البعد (Keras) ودوال المقاييس.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

# ---------- الإعدادات الثابتة (حُدّدت قبل رؤية أي نتيجة) ----------
SEED = 2026
LABELS = [1, 2, 3, 4, 5, 6]
ROOT = "data/raw/uci_har/UCI HAR Dataset"
SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]
CNN_SEEDS = [0, 1, 2]                                   # ثلاث بذور عشوائية لتدريب الشبكة
BATCH, MAX_EPOCHS, PATIENCE, LR = 64, 40, 6, 1e-3       # حجم الدفعة، أقصى عدد حقب، صبر الإيقاف المبكر، معدل التعلم
L2 = 5e-5   # قوة تنظيم L2 على أوزان الشبكة

device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"


def activity_names():
    # أسماء الأنشطة الستة من ملف activity_labels.txt
    acts = pd.read_csv(f"{ROOT}/activity_labels.txt", sep=r"\s+", header=None, names=["id", "name"])
    return acts["name"].tolist()


def load_split(split):
    # تُرجع: الخصائص الجاهزة، والتسميات (1 إلى 6)، وأرقام الأشخاص، والإشارات الخام بشكل (نوافذ، 128، 9)
    X = pd.read_csv(f"{ROOT}/{split}/X_{split}.txt", sep=r"\s+", header=None).to_numpy()
    y = pd.read_csv(f"{ROOT}/{split}/y_{split}.txt", header=None)[0].to_numpy()
    s = pd.read_csv(f"{ROOT}/{split}/subject_{split}.txt", header=None)[0].to_numpy()
    sig = np.stack([pd.read_csv(f"{ROOT}/{split}/Inertial Signals/{n}_{split}.txt",
                                sep=r"\s+", header=None).to_numpy() for n in SIGNALS], axis=-1)
    return X, y, s, sig.astype(np.float32)


def load_all():
    # ندمج ملفي التدريب والاختبار الرسميين معاً (الأشخاص الثلاثون كلهم، وصفوف التدريب أولاً)
    parts = [load_split("train"), load_split("test")]
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    s = np.concatenate([p[2] for p in parts])
    sig = np.concatenate([p[3] for p in parts])
    return X, y, s, sig


def build_cnn(n_steps=128, n_ch=9, n_classes=6):
    # ثلاث كتل تلافيفية (32 و64 و64 مرشحاً)، ثم متوسط شامل، ثم Dropout بنسبة 0.3
    reg = keras.regularizers.l2(L2)
    bn = lambda: keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
    return keras.Sequential([
        keras.Input(shape=(n_steps, n_ch)),
        keras.layers.Conv1D(32, 5, padding="same", kernel_regularizer=reg), bn(), keras.layers.ReLU(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 5, padding="same", kernel_regularizer=reg), bn(), keras.layers.ReLU(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 3, padding="same", kernel_regularizer=reg), bn(), keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, kernel_regularizer=reg),
    ])


class BestMacroF1(keras.callbacks.Callback):
    # إيقاف مبكر يعتمد على macro-F1 في مجموعة التحقق، مع الاحتفاظ بأوزان أفضل حقبة
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val
        self.best_f1, self.best_weights, self.bad, self.epochs_run = -1.0, None, 0, 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_run = epoch + 1
        pred = self.model.predict(self.X_val, batch_size=1024, verbose=0).argmax(1)
        f1 = f1_score(self.y_val, pred, average="macro")
        if f1 > self.best_f1 + 1e-4:
            # تحسّن الأداء: نحفظ الأوزان ونصفّر عدّاد الصبر
            self.best_f1, self.bad = f1, 0
            self.best_weights = self.model.get_weights()
        else:
            # لا تحسّن: نزيد العدّاد، ونوقف التدريب إذا انتهى الصبر
            self.bad += 1
            if self.bad >= PATIENCE:
                self.model.stop_training = True


def predict_cnn(model, X):
    # الفئة المتوقعة هي أعلى قيمة في مخرجات الشبكة
    return model.predict(X, batch_size=1024, verbose=0).argmax(1)


def train_cnn(X_fit, y_fit, X_val, y_val, seed):
    # التسميات هنا من 0 إلى 5. الإيقاف المبكر يستخدم مجموعة التحقق فقط (وليس الاختبار)
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = build_cnn()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    cb = BestMacroF1(X_val, y_val)
    model.fit(X_fit, y_fit, batch_size=BATCH, epochs=MAX_EPOCHS, shuffle=True, verbose=0, callbacks=[cb])
    model.set_weights(cb.best_weights)
    return model, cb.best_f1, cb.epochs_run


def run_cnn(sig_fit, y_fit, sig_val, y_val, sig_test, seed):
    # مصفوفات y بتسميات 1 إلى 6. إحصاءات القنوات (المتوسط والانحراف) تُحسب من نوافذ التدريب فقط
    mu = sig_fit.mean(axis=(0, 1), keepdims=True)
    sd = sig_fit.std(axis=(0, 1), keepdims=True) + 1e-8
    f = lambda a: ((a - mu) / sd).astype(np.float32)
    model, best_f1, n_ep = train_cnn(f(sig_fit), (y_fit - 1).astype(np.int64),
                                     f(sig_val), (y_val - 1).astype(np.int64), seed)
    return predict_cnn(model, f(sig_test)) + 1, n_ep, best_f1


def cm_by_subject(y, p, groups, gids):
    # مصفوفة التباس منفصلة لكل شخص
    from sklearn.metrics import confusion_matrix
    out = np.zeros((len(gids), 6, 6), dtype=np.int64)
    for i, g in enumerate(gids):
        m = groups == g
        out[i] = confusion_matrix(y[m], p[m], labels=LABELS)
    return out


def macro_f1_from_cm(cm):
    # macro-F1 من مصفوفة التباس (الصفوف الحقيقة والأعمدة التوقع)؛ تعمل على مصفوفة واحدة أو مجموعة مصفوفات
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    fp = cm.sum(axis=-2) - tp
    fn = cm.sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean(axis=-1)


def accuracy_from_cm(cm):
    # الدقة = مجموع القطر ÷ المجموع الكلي
    return np.diagonal(cm, axis1=-2, axis2=-1).sum(axis=-1) / cm.sum(axis=(-2, -1))
