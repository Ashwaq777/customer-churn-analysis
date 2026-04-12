import pandas as pd
import matplotlib.pyplot as plt
import os

# ── 1. CHARGEMENT ──────────────────────────────────────────
df = pd.read_csv("data_customer.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())

# ── 2. NETTOYAGE ───────────────────────────────────────────
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
print("Dataset propre :", df.shape)
print("Taux de churn global :", round(df["Churn"].mean() * 100, 2), "%")

# ── 3. ANALYSE ─────────────────────────────────────────────
print(df.groupby("Contract")["Churn"].mean().sort_values(ascending=False))
print(df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False))
print(df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False))
print(df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]].mean())

# ── 4. VISUALISATIONS ──────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs("figures", exist_ok=True)

def plot_churn(groupby_col, title, colors, filename):
    data = df.groupby(groupby_col)["Churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(data.index, data.values, color=colors,
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, data.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.0%}", ha="center", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Taux de churn", fontsize=12)
    ax.set_ylim(0, data.max() + 0.15)
    ax.tick_params(axis="x", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}.png", dpi=150, bbox_inches="tight")

plot_churn("Contract", "Churn par type de contrat",
           ["#e74c3c", "#e67e22", "#2ecc71"], "churn_contract")

plot_churn("PaymentMethod", "Churn par méthode de paiement",
           ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"], "churn_payment")

plot_churn("InternetService", "Churn par service internet",
           ["#e74c3c", "#e67e22", "#2ecc71"], "churn_internet")

# ── 5. TENURE DISTRIBUTION ─────────────────────────────────
churned = df[df["Churn"] == 1]["tenure"]
retained = df[df["Churn"] == 0]["tenure"]

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(retained, bins=30, alpha=0.6, color="#3498db", label="Retained", zorder=1)
ax.hist(churned,  bins=30, alpha=0.85, color="#e74c3c", label="Churned",  zorder=2)
ax.set_title("Distribution de l'ancienneté : Churned vs Retained",
             fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("Tenure (mois)", fontsize=12)
ax.set_ylabel("Nombre de clients", fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axvline(x=12, color="#2c3e50", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(13, ax.get_ylim()[1] * 0.85, "Seuil critique\n12 mois",
        fontsize=10, color="#2c3e50", fontweight="bold")
ax.legend(fontsize=12, framealpha=0.5)
plt.tight_layout()
plt.savefig("figures/tenure_distribution.png", dpi=150, bbox_inches="tight")

plt.show()