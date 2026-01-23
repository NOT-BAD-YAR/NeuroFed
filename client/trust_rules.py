import pandas as pd
import numpy as np
from scipy.stats import chi2

class ResearchTrustLayer:
    def __init__(self):
        self.mean = None
        self.inv_cov = None
        self.reasons = []

    def _fit_distribution(self, df):
        """Learns the global correlation of healthcare features."""
        numeric_df = df.select_dtypes(include=[np.number])
        self.mean = np.mean(numeric_df, axis=0)
        cov = np.cov(numeric_df.T) + np.eye(numeric_df.shape[1]) * 1e-6
        self.inv_cov = np.linalg.inv(cov)

    def symbolic_check(self, row):
        """Hard constraints: Physiological impossibilities."""
        if not (0 <= row['age'] <= 120): return "Invalid Age"
        if row['heart_rate'] < 30 or row['heart_rate'] > 220: return "Lethal HR"
        return None

    def filter_data(self, df):
        self.reasons = []
        if self.mean is None: self._fit_distribution(df)
        
        numeric_df = df.select_dtypes(include=[np.number])
        valid_rows, trust_scores = [], []

        for i, (idx, row) in enumerate(df.iterrows()):
            # 1. Symbolic Filter
            violation = self.symbolic_check(row)
            if violation:
                self.reasons.append({**row.to_dict(), "reason": violation})
                continue

            # 2. Neuro-Statistical Filter (Mahalanobis Distance)
            diff = numeric_df.iloc[i] - self.mean
            m_dist_sq = np.dot(np.dot(diff, self.inv_cov), diff.T)
            p_val = 1 - chi2.cdf(m_dist_sq, df=numeric_df.shape[1])

            if p_val < 0.001: # Statistically impossible correlation
                self.reasons.append({**row.to_dict(), "reason": f"Anomaly (p={p_val:.4f})"})
            else:
                valid_rows.append(row)
                # Trust score based on 'closeness' to the clinical norm
                trust_scores.append(max(0.1, min(1.0, p_val * 5)))

        return pd.DataFrame(valid_rows), pd.DataFrame(self.reasons), np.array(trust_scores)