import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
# Assuming the dataset is stored in a CSV file
df = pd.read_csv('D:/DPARG/Darpg_P2/train.csv')

# Encode labels
label_encoder = LabelEncoder()
df['encoded_category'] = label_encoder.fit_transform(df['parent_name'])

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['summarized_text'], df['encoded_category'], 
                                                                    random_state=42, test_size=0.2)

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# Tokenize the texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              torch.tensor(train_labels.tolist()))

val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
                            torch.tensor(val_encodings['attention_mask']),
                            torch.tensor(val_labels.tolist()))

# Define the batch size and create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(df['parent_name'].unique()))

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(3):  # Train for 3 epochs
    model.train()
    total_train_loss = 0
    
    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        model.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    # Calculate average training loss
    avg_train_loss = total_train_loss / len(train_loader)
    print(f'Epoch {epoch + 1}:')
    print(f'Training Loss: {avg_train_loss:.4f}')
    
    # Validation loop
    model.eval()
    total_val_accuracy = 0
    
    for batch in val_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        total_val_accuracy += accuracy
    
    avg_val_accuracy = total_val_accuracy / len(val_loader)
    print(f'Validation Accuracy: {avg_val_accuracy:.4f}')
