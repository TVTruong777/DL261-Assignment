from google.colab import drive
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import time

drive.mount('/content/drive')

def fit_compile(model, loss_fn, optimizer, epochs, max_patience, train_loader, val_loader, save_path):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  batch_acc_record = []
  val_accuracy_record = []
  best_val_acc = 0.0
  patience = 0
  start_time = time.time()
  for epoch in range(epochs):
    model.train()
    for batchidx, (images, labels) in enumerate(train_loader):
      images = images.to(device)
      labels = labels.to(device)
      predictions = model(images)
      loss = loss_fn(predictions, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if batchidx % 100 == 0:
        print(f"Current epoch: {epoch}, current batch: {batchidx}, current loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
      correct_train = 0
      total_train = 0
      for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        correct_train += (predictions.argmax(dim = 1) == labels).sum().item()
        total_train += labels.size(0)
      train_acc = (correct_train / total_train) * 100
      batch_acc_record.append(train_acc)
      print(f"Epoch: {epoch}, Current training accuracy: {train_acc:.4f}%")

      correct_val = 0
      total_val = 0
      for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        correct_val += (predictions.argmax(dim = 1) == labels).sum().item()
        total_val += labels.size(0)
      val_acc = (correct_val / total_val) * 100
      print(f"Epoch: {epoch}, Current validation accuracy: {val_acc:.4f}%")
      val_accuracy_record.append(val_acc)

      if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = 0
        torch.save(model.state_dict(), save_path)
        print(f"New best model is saved at epoch {epoch}")
      else:
        patience += 1
      if patience >= max_patience:
        print(f"Patience exceeded, training is stopped.")
        break
  end_time = time.time()
  train_time = int(end_time - start_time)
  model.load_state_dict(torch.load(save_path, weights_only = True))
  print(f"Total training time: {train_time} seconds")
  return model, batch_acc_record, val_accuracy_record, train_time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model_metrics(best_models, loader):
  pred_list = []
  label_list = []
  model_result = {}
  for model_name, info in best_models.items():
    model = info[0]
    train_time = info[3]
    start_time = time.perf_counter()

    #inference
    model.eval()
    with torch.no_grad():
      for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim = 1)
        pred_list.extend(preds.cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    #inference time
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time - start_time

    #total parameters
    total_param = sum(p.numel() for p in model.parameters())

    #confusion matrix
    cm = confusion_matrix(y_true = label_list, y_pred = pred_list)
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues' )
    plt.xlabel("Predictions")
    plt.ylabel("Actual class")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    #report metrics
    printed_report = classification_report(y_true = label_list, y_pred = pred_list, digits = 4)
    result_dict = classification_report(y_true = label_list, y_pred = pred_list, digits = 4, output_dict = True)
    print(f"\n----------------------- {model_name} Metrics Report -----------------------\n")
    print(printed_report)
    model_result[model_name] = (result_dict['accuracy'], result_dict['macro avg']['f1-score'], result_dict['weighted avg']['f1-score'], train_time, inference_time, total_param)
  return model_result

def plot_models_curve(best_models):
  for model, info in best_models.items():
    train_track = info[1]
    val_track = info[2]
    plt.plot(train_track, label = "Training", color = "blue")
    plt.plot(val_track, label = "Validation", color = "orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f" {model} - Training vs Validation Accuracy Curve")
    plt.grid(axis = 'both', alpha = 0.5)
    plt.legend()
    plt.show()