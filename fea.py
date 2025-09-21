import numpy as np

def analyze(nodes, members, loads, supports):
    print("🔍 Analiz başlatılıyor...")
    print(f"Düğüm sayısı: {len(nodes)}")
    print(f"Eleman sayısı: {len(members)}")
    print(f"Yüklenen düğüm sayısı: {len(loads)}")
    print(f"Sabitlenen düğüm sayısı: {len(supports)}")

    # Henüz sahte analiz: Her düğüm 0 yer değiştirmiş gibi
    displacements = np.zeros((len(nodes), 3))

    return displacements
