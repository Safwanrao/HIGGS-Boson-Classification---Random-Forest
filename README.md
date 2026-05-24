# ── Graph 1: Confusion Matrix ──
fig, ax = plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
    display_labels=['Background','Signal']).plot(ax=ax, cmap='Blues')
ax.set_title('Confusion Matrix — Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/1_confusion_matrix.png', dpi=150)
plt.show()

<img width="593" height="453" alt="image" src="https://github.com/user-attachments/assets/64f1a090-87f2-45df-8d85-486e20e2deb8" />

# ── Graph 2: ROC Curve ──
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(fpr, tpr, color='#e63946', lw=2, label=f'AUC = {roc_auc:.4f}')
ax.plot([0,1],[0,1], 'grey', linestyle='--')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Random Forest', fontsize=14, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/2_roc_curve.png', dpi=150)
plt.show()

<img width="690" height="490" alt="image" src="https://github.com/user-attachments/assets/adf95430-547c-4e1e-a5f5-8cae3470f141" />

# ── Graph 3: Feature Importances ──
feat_names = df.drop(columns=['label']).columns.tolist()
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:15]

fig, ax = plt.subplots(figsize=(10,6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, 15))
ax.bar(range(15), importances[indices], color=colors)
ax.set_xticks(range(15))
ax.set_xticklabels([feat_names[i] for i in indices], rotation=45, ha='right', fontsize=9)
ax.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
ax.set_ylabel('Importance Score'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/3_feature_importances.png', dpi=150)
plt.show()

# ── Graph 4: Class Distribution ──
fig, axes = plt.subplots(1, 2, figsize=(10,4))
pd.Series(y_test).value_counts().sort_index().plot(
    kind='bar', ax=axes[0], color=['#457b9d','#e63946'], rot=0)
axes[0].set_title('Actual Classes', fontweight='bold')
axes[0].set_xticklabels(['Background','Signal'])

pd.Series(y_pred).value_counts().sort_index().plot(
    kind='bar', ax=axes[1], color=['#2a9d8f','#f4a261'], rot=0)
axes[1].set_title('Predicted Classes', fontweight='bold')
axes[1].set_xticklabels(['Background','Signal'])

plt.suptitle('Actual vs Predicted Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/4_class_distribution.png', dpi=150)
plt.show()

# ── Graph 5: Probability Distribution ──
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(y_proba[y_test==0], bins=60, alpha=0.6, color='#457b9d', label='Background')
ax.hist(y_proba[y_test==1], bins=60, alpha=0.6, color='#e63946', label='Signal')
ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Count')
ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/5_probability_distribution.png', dpi=150)
plt.show()
print('✅ Graph 5 save ho gaya!')
print('\n🎉 Saare 5 graphs ban gaye results/ folder mein!')

<img width="789" height="490" alt="image" src="https://github.com/user-attachments/assets/c747a55d-3a62-4567-8d7c-7c0cd2e50f8a" />
