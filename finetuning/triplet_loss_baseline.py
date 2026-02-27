import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from transformers import EarlyStoppingCallback

# Parameters — identical to the paper
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
train_csv = '/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/data/clusters/clusters_train.csv'
val_csv = '/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/data/clusters/clusters_val.csv'
output_dir = '/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/models/triplet_model'
batch_size = 8
epochs = 50
learning_rate = 2e-5

# Load and prepare data
train_df = pd.read_csv(train_csv, low_memory=False)
val_df = pd.read_csv(val_csv, low_memory=False)

train_df['Cluster ID'] = train_df['Cluster ID'].astype(int)
val_df['Cluster ID'] = val_df['Cluster ID'].astype(int)

# Mapping of labels to sequential mapping
unique_train_labels = sorted(train_df['Cluster ID'].unique())
label_mapping = {orig_label: new_label for new_label, orig_label in enumerate(unique_train_labels)}

train_df['Mapped Label'] = train_df['Cluster ID'].map(label_mapping)
val_df = val_df[val_df['Cluster ID'].isin(label_mapping.keys())].copy()
val_df['Mapped Label'] = val_df['Cluster ID'].map(label_mapping)

# Create HuggingFace Datasets with columns expected by SentenceTransformerTrainer
# BatchAllTripletLoss expects a "sentence" column and a "label" column
train_dataset = Dataset.from_dict({
    "sentence": [str(t) for t in train_df['Playlist Title'].tolist()],
    "label": train_df['Mapped Label'].tolist(),
})
val_dataset = Dataset.from_dict({
    "sentence": [str(t) for t in val_df['Playlist Title'].tolist()],
    "label": val_df['Mapped Label'].tolist(),
})

print(f"Number of classes: {len(label_mapping)}")
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

model = SentenceTransformer(MODEL_NAME, device=device)

# Loss definition — identical to the paper
train_loss = losses.BatchAllTripletLoss(model=model)

# Training arguments
training_args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    warmup_steps=100,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=2,
    logging_strategy="epoch",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=train_loss,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()

model.save_pretrained(output_dir)
print(f"Model saved to: {output_dir}")
