import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── 1. CHARGEMENT ──────────────────────────────────────────
df = pd.read_csv("data_customer.csv")
print("Shape:", df.shape)
print(df.head())
# ── 2. NETTOYAGE ──────────────────────────────────────────
# Supprimer customerID inutile
df = df.drop(columns=["customerID"])

# TotalCharges en numérique
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Churn en binaire
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ── 3. ENCODAGE ────────────────────────────────────────────
# Convertir toutes les colonnes texte en chiffres
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

print("Dataset prêt :", df.shape)
print(df.head())
# ── 4. TRAIN/TEST SPLIT ────────────────────────────────────
X = df.drop(columns=["Churn"])  # Features
y = df["Churn"]                  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ── 5. ENTRAÎNER LE MODÈLE ─────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Modèle entraîné !")

# ── 6. PRÉDICTIONS ─────────────────────────────────────────
y_pred = model.predict(X_test)

# ── 7. RÉSULTATS ───────────────────────────────────────────
print("\n📊 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
# ── 8. FEATURE IMPORTANCE ──────────────────────────────────
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(10, 8))
colors = ['#e74c3c' if i >= len(importances)-3 else '#3498db' 
          for i in range(len(importances))]
importances.plot(kind='barh', color=colors)
plt.title("Feature Importance — What drives Churn ?", 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Importance Score", fontsize=12)
plt.axvline(x=importances.mean(), color='gray', 
            linestyle='--', alpha=0.7, label='Mean importance')
plt.legend()
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150, bbox_inches='tight')

# ── 9. CONFUSION MATRIX ────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title("Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.xticks([0, 1], ['Retained', 'Churned'])
plt.yticks([0, 1], ['Retained', 'Churned'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                fontsize=20, fontweight='bold',
                color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150, bbox_inches='tight')

plt.show()
print("✅ Graphiques sauvegardés dans figures/")