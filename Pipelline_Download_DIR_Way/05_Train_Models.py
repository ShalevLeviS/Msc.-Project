import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, DenseNet121, InceptionV3
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import shutil
import tensorflow as tf
from PIL import Image

# Fix PIL decompression bomb error for large images
Image.MAX_IMAGE_PIXELS = None  # Remove the limit entirely
# Or set a higher limit: Image.MAX_IMAGE_PIXELS = 1000000000

# === Load & filter data ===
df = pd.read_csv("/local1/ofir/shalevle/STImage_1K4M/merged_cleaned.csv")
df = df.dropna(subset=['top_subtype'])
df = df[df['top_subtype'].str.lower() != 'normal']

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['top_subtype'], random_state=42)

# === Save images into folders (with smart preprocessing) ===
def smart_preprocess_image(src_path, dst_path, target_megapixels=16):
    """
    Smart resizing based on your data analysis:
    - Small images (<=10 MP): Keep original size
    - Large images (>10 MP): Resize to ~16 MP maintaining aspect ratio
    This gives much better quality than brutal 224x224 downsampling
    """
    try:
        with Image.open(src_path) as img:
            current_mp = (img.width * img.height) / 1_000_000
            
            print(f"Processing {os.path.basename(src_path)}: {img.width}x{img.height} ({current_mp:.1f} MP)", end="")
            
            if current_mp > 10:  # Only resize the large images
                # Calculate scale to reach target megapixels
                scale_factor = (target_megapixels / current_mp) ** 0.5
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_resized.save(dst_path, quality=95)  # High quality JPEG
                
                new_mp = (new_width * new_height) / 1_000_000
                print(f" → {new_width}x{new_height} ({new_mp:.1f} MP)")
            else:
                # Small images: copy as-is
                shutil.copy(src_path, dst_path)
                print(f" → kept original size")
                
    except Exception as e:
        print(f"\nError processing {src_path}: {e}")
        # Fallback: try to copy original
        try:
            shutil.copy(src_path, dst_path)
            print("Fallback: copied original")
        except:
            print(f"Failed to copy {src_path}")

base_dir = "cnn_data"
print("Creating directory structure and preprocessing images...")

for split_name, split_df in [('train', train_df), ('val', val_df)]:
    print(f"\n=== Processing {split_name.upper()} set ({len(split_df)} images) ===")
    
    # Create directories
    for label in df['top_subtype'].unique():
        os.makedirs(f"{base_dir}/{split_name}/{label}", exist_ok=True)
    
    # Process images
    for i, (_, row) in enumerate(split_df.iterrows(), 1):
        src = row['image_path']
        dst = f"{base_dir}/{split_name}/{row['top_subtype']}/{os.path.basename(src)}"
        print(f"[{i}/{len(split_df)}] ", end="")
        smart_preprocess_image(src, dst)

# === Data generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory("cnn_data/train", target_size=(224, 224), batch_size=32, class_mode='categorical')
val_gen = val_datagen.flow_from_directory("cnn_data/val", target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# === Compute class weights ===
class_labels = list(train_gen.class_indices.keys())
# Fix: Convert class_labels to numpy array
weights = compute_class_weight(class_weight="balanced", classes=np.array(class_labels), y=train_df['top_subtype'])
class_weight = {i: weights[i] for i in range(len(class_labels))}

# === Model builder ===
def build_model(base_model, num_classes):
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# === Train and evaluate all models ===
results = {}
model_dict = {
    "VGG16": VGG16,
    "ResNet50": ResNet50,
    "EfficientNetB0": EfficientNetB0,
    "DenseNet121": DenseNet121,
    "InceptionV3": InceptionV3
}

for name, base_fn in model_dict.items():
    print(f"\n🔁 Training: {name}")
    base = base_fn(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = build_model(base, len(class_labels))
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=10, class_weight=class_weight,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)

    y_true = val_gen.classes
    y_pred = model.predict(val_gen)
    y_pred_labels = np.argmax(y_pred, axis=1)

    report = classification_report(y_true, y_pred_labels, target_names=class_labels, output_dict=True)
    auc = roc_auc_score(tf.keras.utils.to_categorical(y_true), y_pred, multi_class='ovr')

    results[name] = {
        "accuracy": report["accuracy"],
        "macro avg f1": report["macro avg"]["f1-score"],
        "AUC": auc
    }

# === Plot comparison ===
df_results = pd.DataFrame(results).T
df_results.to_csv("/local1/ofir/shalevle/STImage_1K4M/resukt_models_comparsion.csv", index=True)








