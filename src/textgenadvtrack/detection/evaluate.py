from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def detection_metrics(labels, human_scores) -> dict[str, float]:
    preds = [1 if score >= 0.5 else 0 for score in human_scores]
    auc = roc_auc_score(labels, human_scores)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    final_score = (0.6 * auc + 0.3 * acc + 0.1 * f1) / 100.0
    return {"AUC": auc, "ACC": acc, "F1": f1, "Final_Score": final_score}
