import numpy as np
from sklearn.linear_model import LinearRegression


def inference_attack(noisy_data, public_data, threshold=0.75):
    record_count = noisy_data.shape[0]
    dim_count = noisy_data.shape[1]

    corr = np.corrcoef(public_data, rowvar=False)
    final_result = []
    # Attack dimension i
    for i in range(dim_count):
        noisy_i = noisy_data[:, i]
        infer_i = []
        for j in range(dim_count):
            if i == j:
                continue
            corr_ij = corr[i][j]
            # Find highly correlated dimensions
            if (threshold <= corr_ij <= 1.0) or (-1.0 <= corr_ij <= -threshold):
                noisy_j = noisy_data[:, j]
                model = LinearRegression()
                X = public_data[:, j].reshape(-1, 1)
                Y = public_data[:, i].reshape(-1)
                model.fit(X, Y)
                alpha = model.coef_[0]
                beta = model.intercept_
                infer_i_by_j = alpha * noisy_j + beta
                infer_i.append((corr_ij, infer_i_by_j))
        final_i = np.zeros(record_count) + noisy_i
        weights_i = 1
        # Weighted average
        for k in range(len(infer_i)):
            final_i += infer_i[k][0] * infer_i[k][0] * infer_i[k][1]
            weights_i += infer_i[k][0] * infer_i[k][0]
        final_i /= weights_i
        final_result.append(final_i)
    final_result = np.array(final_result).transpose(1, 0)
    return final_result
