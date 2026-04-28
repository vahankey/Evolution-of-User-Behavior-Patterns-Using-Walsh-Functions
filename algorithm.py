import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans


def fwht(data: list) -> list:
    n = len(data)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = data[j]
                y = data[j + h]
                data[j] = x + y
                data[j + h] = x - y
        h *= 2
    return data


def pad_to_power_of_2(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    power = 1
    while power < n:
        power *= 2
    return np.pad(arr, (0, power - n), "constant")


def extract_spectrum(time_series: np.ndarray) -> np.ndarray:
    """Return L1-normalised FWHT spectrum of a time series."""
    padded = pad_to_power_of_2(np.asarray(time_series, dtype=float))
    spectrum = fwht(list(padded))
    n = len(spectrum)
    result = np.array(spectrum) / n

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result




rng = np.random.default_rng(42)  # reproducible random state

normal_users_logs = [rng.poisson(lam=2, size=16) for _ in range(100)]

bot_logs = []
for _ in range(3):
    base_bot = np.array([20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0])
    noise = rng.integers(0, 3, 16)
    bot_logs.append(base_bot + noise)

all_users_logs = normal_users_logs + bot_logs  # indices 0-99 normal, 100-102 bots

X_features = np.array([extract_spectrum(logs) for logs in all_users_logs])



print("=" * 55)
print("  ԲՈՏԵՐԻ ԲԱՑԱՀԱՅՏՈՒՄ (ISOLATION FOREST)")
print("=" * 55)

iso_forest = IsolationForest(contamination=0.04, random_state=42)
predictions = iso_forest.fit_predict(X_features)

tp = fp = 0
for i, pred in enumerate(predictions):
    if pred == -1:  # flagged as anomaly
        is_actual_bot = i >= 100
        label = "ԲՈՏ ✓ (ճիշտ)" if is_actual_bot else "ՆՈՐՄԱԼ ✗ (կեղծ ահազանգ)"
        status = "TRUE POSITIVE" if is_actual_bot else "FALSE POSITIVE"
        if is_actual_bot:
            tp += 1
        else:
            fp += 1
        print(f"\n[{status}] Օգտատեր ID {i:>3} → {label}")
        print(f"  Սպեկտր: {np.round(X_features[i], 2)}")

print(f"\nԱրդյունք: {tp} ճիշտ բոտ, {fp} կեղծ ահազանգ")



print("\n" + "=" * 55)
print("  ՆՈՐՄԱԼ ՎԱՐՔԱԳԾԻ ԽՄԲԱՎՈՐՈՒՄ (K-MEANS)")
print("=" * 55)

normal_features = X_features[:100]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(normal_features)

centroids = kmeans.cluster_centers_

centroid_energies = [np.mean(np.abs(c)) for c in centroids]

sorted_indices = np.argsort(centroid_energies)
activity_labels = {
    sorted_indices[0]: "Պասիվ Ընթերցողներ",
    sorted_indices[1]: "Միջին Ակտիվություն",
    sorted_indices[2]: "Ակտիվ Օգտատերեր",
}

cluster_counts = np.bincount(kmeans.labels_, minlength=3)

print("\nԽՄԲԵՐԻ ՎԵՐԾԱՆՈՒՄ")
print("-" * 45)
for i, center in enumerate(centroids):
    energy = centroid_energies[i]
    name = activity_labels[i]
    count = cluster_counts[i]
    print(f"Խումբ {i}: {name}")
    print(f"  Անդամներ: {count}  |  Միջին սպեկտրալ էներգիա: {energy:.4f}")



print("\n" + "=" * 55)
print("  ՎԱՐՔԱԳԾԱՅԻՆ ՇԵՂՄԱՆ ՄՈՆԻԹՈՐԻՆԳ (ACCOUNT TAKEOVER)")
print("=" * 55)


def detect_drift(
    window1: np.ndarray,
    window2: np.ndarray,
    threshold: float = 2.0,
    label: str = "",
) -> float:
    """
    Compare two time windows via their FWHT spectra.
    Returns the Euclidean distance between the two spectra.
    """
    spec1 = extract_spectrum(window1)
    spec2 = extract_spectrum(window2)
    distance = np.linalg.norm(spec1 - spec2)

    verdict = (
        "⚠  ՖԻՔՍՎԵԼ Է ԿՏՐՈՒԿ ՇԵՂՈՒՄ (հնարավոր պրոֆիլի գողություն)"
        if distance > threshold
        else "✓  Վարքագիծը նորմալ էվոլյուցիայի սահմաններում է"
    )

    print(f"\n{'─'*45}")
    if label:
        print(f"Դեպք: {label}")
    print(f"Սպեկտրալ հեռավորություն (ΔD): {distance:.4f}  (շեմ={threshold})")
    print(verdict)
    return distance



detect_drift(
    np.array([1, 2, 1, 1, 2, 1, 1, 0]),
    np.array([2, 3, 2, 2, 3, 2, 2, 1]),
    label="Օգտատերը պարզապես սկսեց ավելի ակտիվ կարդալ",
)


detect_drift(
    np.array([2, 3, 2, 2, 3, 2, 2, 1]),
    np.array([8, 0, 8, 0, 8, 0, 8, 0]),
    label="Սպամ բոտ — կտրուկ արհեստական ռիթմ",
)
