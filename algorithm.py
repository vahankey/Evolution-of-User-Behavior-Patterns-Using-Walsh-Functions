import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

def fwht(data):
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

def pad_to_power_of_2(arr):
    n = len(arr)
    power = 1
    while power < n:
        power *= 2

    return np.pad(arr, (0, power - n), 'constant')

def extract_spectrum(time_series):
    padded_data = pad_to_power_of_2(time_series)
    data_copy = list(padded_data)
    spectrum = fwht(data_copy)
    
    n = len(spectrum)
    return np.array(spectrum) / n

normal_users_logs = [np.random.poisson(lam=2, size=16) for _ in range(100)]



bot_logs = []
for _ in range(3):
    base_bot = np.array([20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0])
    noise = np.random.randint(0, 3, 16) 
    bot_logs.append(base_bot + noise)


all_users_logs = normal_users_logs + bot_logs


X_features = np.array([extract_spectrum(logs) for logs in all_users_logs])



print("--- ԲՈՏԵՐԻ ԲԱՑԱՀԱՅՏՈՒՄ (ISOLATION FOREST) ---")


iso_forest = IsolationForest(contamination=0.04, random_state=42)
predictions = iso_forest.fit_predict(X_features)


for i, pred in enumerate(predictions):
    if pred == -1: 
        user_type = "ՆՈՐՄԱԼ" if i < 100 else "ԲՈՏ"
        print(f"Ահազանգ! Օգտատեր ID {i} ճանաչվել է որպես անոմալիա: Իրականում նա՝ {user_type} է:")
        print(f"  Նրա Սպեկտրը: {np.round(X_features[i], 2)}")

print("\n--- ՆՈՐՄԱԼ ՎԱՐՔԱԳԾԻ ԽՄԲԱՎՈՐՈՒՄ (K-MEANS) ---")

normal_features = X_features[:100]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(normal_features)
print(f"Նորմալ օգտատերերը բաժանվեցին 3 խմբի (օրինակ՝ պասիվ, միջին, ակտիվ):")
print(f"Խմբերի կենտրոնները (Centroids) գտնվում են բազմաչափ սպեկտրալ տարածությունում:\n")

centroids = kmeans.cluster_centers_

print("\n--- ԽՄԲԵՐԻ ՎԵՐԾԱՆՈՒՄ ---")
for i, center in enumerate(centroids):
    
    activity_level = center[0]
    
    if activity_level > 2.0: 
        name = "Ակտիվ Օգտատերեր"
    elif activity_level > 1.0:
        name = "Միջին Ակտիվություն"
    else:
        name = "Պասիվ Ընթերցողներ"
        
    print(f"Խումբ {i}: {name} (Միջին էներգիա X_0 = {activity_level:.2f})")


print("\n--- ՎԱՐՔԱԳԾԱՅԻՆ ՇԵՂՄԱՆ ՄՈՆԻԹՈՐԻՆԳ (ACCOUNT TAKEOVER) ---")

def detect_drift(window1, window2, threshold=2.0):
    """Հաշվում է Եվկլիդեսյան հեռավորությունը երկու պատուհանների սպեկտրների միջև"""
    spec1 = extract_spectrum(window1)
    spec2 = extract_spectrum(window2)
    
    
    distance = np.linalg.norm(spec1 - spec2)
    
    print(f"Սպեկտրալ հեռավորություն (ΔD): {distance:.2f}")
    if distance > threshold:
        print("Ֆիքսվել է վարքագծի կտրուկ շեղում (Հնարավոր պրոֆիլի գողություն):")
    else:
        print("Վարքագիծը նորմալ էվոլյուցիայի սահմաններում է:")

print("\nԴեպք Ա. Օգտատերը պարզապես սկսեց մի քիչ ավելի ակտիվ կարդալ նյութեր")
window_t1 = np.array([1, 2, 1, 1, 2, 1, 1, 0])
window_t2 = np.array([2, 3, 2, 2, 3, 2, 2, 1])
detect_drift(window_t1, window_t2)

print("\nԴեպք Բ. սպամ բոտ (Կտրուկ արհեստական ռիթմ)")
window_t3 = np.array([2, 3, 2, 2, 3, 2, 2, 1]) 
window_t4 = np.array([8, 0, 8, 0, 8, 0, 8, 0]) 
detect_drift(window_t3, window_t4)
