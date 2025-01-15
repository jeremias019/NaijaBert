# Evaluate NaijaBERT
naijabert_predictions = []
true_labels = []
model.eval()

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        naijabert_predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())

# Save results
test_data['NaijaBERT_Predicted_Label'] = naijabert_predictions

# Classification reports
print("NaijaBERT Report:")
print(classification_report(true_labels, naijabert_predictions))
print("VADER Report:")
print(classification_report(test_data['Label'], test_data['VADER_Predicted_Label']))
print("Logistic Regression Report:")
print(classification_report(test_data['Label'], test_data['Logistic_Predicted_Label']))

# Save reports to drive
classification_report_path = '/content/drive/MyDrive/JeremyDissertation/classification_report_naijabert.txt'
with open(classification_report_path, 'w') as file:
    file.write(classification_report(true_labels, naijabert_predictions))
