import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos del experimento
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("multimodal_fake_news_detection_v1")
runs = client.search_runs(experiment.experiment_id)

# Convertir a DataFrame
df = pd.DataFrame([run.data.metrics for run in runs])

# Graficar
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(df['train_accuracy'], label='Train')
plt.plot(df['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['train_loss'], label='Train')
plt.plot(df['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



'''

📸 Ejemplo de cómo debería verse
Si quieres verificar que todo funcionó, ejecuta este script:

import mlflow
from mlflow.tracking import MlflowClient

# Configurar
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Obtener información del run
client = MlflowClient()
experiment = client.get_experiment_by_name("multimodal_fake_news")

if experiment:
    runs = client.search_runs(experiment.experiment_id)
    
    for run in runs:
        print(f"\n{'='*60}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
        print(f"Status: {run.info.status}")
        print(f"Start: {run.info.start_time}")
        
        print(f"\n📊 PARÁMETROS:")
        for key, value in run.data.params.items():
            print(f"  {key}: {value}")
        
        print(f"\n📈 MÉTRICAS FINALES:")
        for key, value in run.data.metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\n🎯 ARTIFACTS:")
        artifacts = client.list_artifacts(run.info.run_id)
        for artifact in artifacts:
            print(f"  - {artifact.path}")
else:
    print("❌ No se encontró el experimento 'multimodal_fake_news'")
'''


'''

🎨 Crear visualizaciones personalizadas
Si quieres ver las gráficas fuera del UI:

import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Cargar métricas
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("multimodal_fake_news")
runs = client.search_runs(experiment.experiment_id)

# Obtener el último run
latest_run = runs[0]
run_id = latest_run.info.run_id

# Obtener historial de métricas
def get_metric_history(run_id, metric_name):
    metric_history = client.get_metric_history(run_id, metric_name)
    return [(m.step, m.value) for m in metric_history]

# Crear gráficas
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
train_acc = get_metric_history(run_id, "train_accuracy")
val_acc = get_metric_history(run_id, "val_accuracy")
axes[0, 0].plot([x[0] for x in train_acc], [x[1] for x in train_acc], 'b-', label='Train')
axes[0, 0].plot([x[0] for x in val_acc], [x[1] for x in val_acc], 'r-', label='Val')
axes[0, 0].set_title('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Loss
train_loss = get_metric_history(run_id, "train_loss")
val_loss = get_metric_history(run_id, "val_loss")
axes[0, 1].plot([x[0] for x in train_loss], [x[1] for x in train_loss], 'b-', label='Train')
axes[0, 1].plot([x[0] for x in val_loss], [x[1] for x in val_loss], 'r-', label='Val')
axes[0, 1].set_title('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# F1
train_f1 = get_metric_history(run_id, "train_f1")
val_f1 = get_metric_history(run_id, "val_f1")
axes[1, 0].plot([x[0] for x in train_f1], [x[1] for x in train_f1], 'b-', label='Train')
axes[1, 0].plot([x[0] for x in val_f1], [x[1] for x in val_f1], 'r-', label='Val')
axes[1, 0].set_title('F1 Score')
axes[1, 0].legend()
axes[1, 0].grid(True)

# AUC
train_auc = get_metric_history(run_id, "train_auc")
val_auc = get_metric_history(run_id, "val_auc")
axes[1, 1].plot([x[0] for x in train_auc], [x[1] for x in train_auc], 'b-', label='Train')
axes[1, 1].plot([x[0] for x in val_auc], [x[1] for x in val_auc], 'r-', label='Val')
axes[1, 1].set_title('AUC-ROC')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('all_metrics.png', dpi=150)
plt.show()

print("✅ Gráficas guardadas en 'all_metrics.png'")
'''


'''
🔧 Comparar múltiples runs
# En MLflow UI:
# 1. Selecciona múltiples runs (checkbox)
# 2. Click en "Compare"
# 3. Verás gráficas paralelas

# O por código:
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("multimodal_fake_news")
runs = client.search_runs(
    experiment.experiment_id,
    order_by=["metrics.val_accuracy DESC"],
    max_results=5
)

# Crear tabla comparativa
import pandas as pd

comparison_data = []
for run in runs:
    comparison_data.append({
        'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
        'learning_rate': run.data.params.get('learning_rate'),
        'batch_size': run.data.params.get('batch_size'),
        'val_accuracy': run.data.metrics.get('best_val_accuracy', 0),
        'val_loss': run.data.metrics.get('best_val_loss', 0),
    })

df_comparison = pd.DataFrame(comparison_data)
print("\n📊 COMPARACIÓN DE RUNS:")
print(df_comparison.to_string(index=False))
'''