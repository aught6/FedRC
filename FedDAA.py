import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import copy
from torch.utils.data import ConcatDataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def get_client_prototype(client, model, num_classes, device='cuda'):

    original_device = next(model.parameters()).device
    model = model.to(device)
    """
    計算 Client k 在時間 t 的 Data Prototype (Eq. 2)
    """
    model.eval()
    
    class_output_sums = {r: torch.zeros(num_classes).to(device) for r in range(num_classes)}
    class_counts = {r: 0 for r in range(num_classes)}
    
    with torch.no_grad():
        # 遍歷該用戶的訓練資料集
        for batch_data in client.train_iterator:
            # 不管 DataLoader 吐出幾個東西，我們永遠只拿第 0 個 (圖片) 和第 1 個 (標籤)
            inputs = batch_data[0].to(device)
            targets = batch_data[1].to(device)
            
            # 取得模型輸出
            outputs = model(inputs) 
            
            # 將每個樣本的輸出累加到對應的類別中
            for i in range(len(targets)):
                y = int(targets[i].item())
                class_output_sums[y] += outputs[i]
                class_counts[y] += 1
                
    # 計算每個類別的平均輸出，並串接
    P_k = []
    for r in range(num_classes):
        if class_counts[r] > 0:
            mean_output = class_output_sums[r] / class_counts[r]
        else:
            # 防呆：避免除以 0
            mean_output = torch.zeros(num_classes).to(device)
            
        P_k.append(mean_output)
    
    # 將 R 個長度為 R 的向量串接起來，形成 1D numpy 陣列
    P_k = torch.cat(P_k).cpu().numpy()
    model = model.to(original_device)
    return P_k

def run_ncd_module(clients, model, num_classes, device='cuda'):
    """
    執行 Algorithm 1: Number of Clusters Determination (NCD) module
    回傳: 最佳分群數 (optimal_C), 最佳群集中心 (optimal_centers)
    """
    print(">> 執行 NCD Module 尋找最佳分群數...")
    
    R_proto = []
    
    # 步驟 3-6: 收集各個 Client 的 Prototype
    for client in clients:
        P_k = get_client_prototype(client, model, num_classes, device)
        R_proto.append(P_k)
        
    R_proto = np.array(R_proto) 
    
    # 步驟 7-11: 伺服器端計算輪廓係數
    # 設定最大群數 M (不能超過客戶總數-1，且最少要有 2 個客戶才能分群)
    max_M = min(10, len(clients) - 1)
    
    if max_M < 2:
        print("   -> 警告：客戶端數量不足以執行 K-means 分群，預設回傳 1 群。")
        return 1, None
        
    best_score = -1
    optimal_C = 2
    optimal_centers = None
    
    for c in range(2, max_M + 1):
        # 使用 K-means 分成 c 群
        kmeans = KMeans(n_clusters=c, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(R_proto)
        
        # 計算平均輪廓係數 (Eq. 12)
        score = silhouette_score(R_proto, cluster_labels)
        print("   -> 嘗試分群數 C = {}: 輪廓係數 = {:.4f}".format(c, score))
        # 步驟 12: 找出最佳分數與對應的 C
        if score > best_score:
            best_score = score
            optimal_C = c
            optimal_centers = kmeans.cluster_centers_
            
    print(f"   -> NCD 模組計算完成！最佳分群數 C^t = {optimal_C} (輪廓係數: {best_score:.4f})")
    
    return optimal_C, optimal_centers


"RDLD MODULE"
def predict_cluster(P, centers):
    """
    Procedure PREDICT CLUSTER (Algorithm 2)
    計算 Prototype P 與所有 cluster centers 的距離 (Eq. 3)
    回傳距離最近的 cluster 索引 (index)
    """
    distances = []
    for center in centers:
        # 假設 Eq. (3) 為歐式距離 (Euclidean distance)
        dist = np.linalg.norm(P - center)
        distances.append(dist)
    
    # 回傳最小距離的索引
    return np.argmin(distances)

def run_rdld_module(clients, model, num_classes, cluster_centers, prev_prototypes, device='cuda'):
    """
    執行 Algorithm 2: Real Drift Local Detection (RDLD) module
    - prev_prototypes: 字典, 紀錄上一輪各個 client 的 prototype (key: client_index, value: P_k^{t-1})
    """
    print(">> 執行 RDLD Module 偵測概念漂移 (Real Drift)...")
    
    K_clean = []
    K_drift = []
    
    # 建立一個新的字典，用來把這次算好的 Prototypes 存起來，留給下一個 Task 使用
    current_prototypes = {} 
    
    for client_idx, client in enumerate(clients):
        # 第 7 行：取得目前的 Prototype P_k^t
        P_k_current = get_client_prototype(client, model, num_classes, device)
        current_prototypes[client_idx] = P_k_current
        
        # 第 6 行：取得前一輪的 Prototype P_k^{t-1}
        # 如果是第一輪 (Task 0)，還沒有上一輪的紀錄，預設全部視為 Clean
        if prev_prototypes is None or client_idx not in prev_prototypes:
            K_clean.append(client)
            continue
            
        P_k_prev = prev_prototypes[client_idx]
        
        # 第 8, 9 行：預測所屬的 Cluster
        c_prev = predict_cluster(P_k_prev, cluster_centers)
        c_current = predict_cluster(P_k_current, cluster_centers)
        
        # 第 10-14 行：判斷是否發生 Real Drift
        if c_prev == c_current:
            K_clean.append(client)  # 維持在原本的群，標記為 clean
        else:
            K_drift.append(client)  # 換群了，發生 drift

    print(f"   -> RDLD 偵測完成：{len(K_clean)} 個維持 Clean，{len(K_drift)} 個發生 Drift。")
    
    # 回傳 drift 清單、clean 清單，以及這次的 prototypes (當作下一輪的 prev_prototypes)
    return K_drift, K_clean, current_prototypes

"FedDAA"
def run_feddaa_adaptation(clients, K_drift, prev_datasets):
    """
    執行 Algorithm 3: 概念漂移資料適應 (第 10-14 行)
    - K_drift: 發生 drift 的用戶列表
    - prev_datasets: 字典，紀錄上一輪各個 client 的資料集 S_k^{t-1}
    回傳: 備份好的當前資料集，供下一輪當作舊資料使用
    """
    print(">> 執行 FedDAA 資料適應：Clean 用戶合併新舊資料，Drift 用戶僅用新資料...")
    
    current_datasets_for_next_round = {}
    
    for client_idx, client in enumerate(clients):
        # 取得目前的 dataset (S_k^t)
        current_dataset = client.train_iterator.dataset
        batch_size = client.train_iterator.batch_size
        
        # 使用 deepcopy 備份當前的 dataset，作為下一輪的 S_k^{t-1}
        # 這樣可以避免下一輪修改標籤或旋轉圖片時，連帶污染到這份舊備份
        snapshot_dataset = copy.deepcopy(current_dataset)
        current_datasets_for_next_round[client_idx] = snapshot_dataset
        
        # 根據演算法 Algorithm 3 進行條件判斷
        if client in K_drift:
            # 第 11 行: Client k experiences real drift -> 只使用 S_k^t
            # 重新封裝 DataLoader 確保乾淨
            client.train_iterator = DataLoader(current_dataset, batch_size=batch_size, shuffle=True)
            
        else:
            # 第 13 行: Client k is clean -> 使用 S_k^t 與 S_k^{t-1} 共同訓練
            if prev_datasets is not None and client_idx in prev_datasets:
                prev_dataset = prev_datasets[client_idx]
                
                # 將舊資料與新資料合併 (ConcatDataset)
                combined_dataset = ConcatDataset([current_dataset, prev_dataset])
                targets_current = current_dataset.targets
                targets_prev = prev_dataset.targets
                
                if isinstance(targets_current, list):
                    combined_dataset.targets = targets_current + targets_prev
                elif isinstance(targets_current, torch.Tensor):
                    combined_dataset.targets = torch.cat((targets_current, targets_prev))
                else: # 假設為 NumPy Array
                    import numpy as np
                    combined_dataset.targets = np.concatenate((targets_current, targets_prev))
                # 更新該用戶的 DataLoader，讓他在這輪訓練時抽到新舊混合的資料
                client.train_iterator = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
            else:
                # 第一輪 (t=1) 還沒有舊資料，所以只用當前資料
                client.train_iterator = DataLoader(current_dataset, batch_size=batch_size, shuffle=True)
                
    return current_datasets_for_next_round