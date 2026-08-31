"""
yolo_multi.py
Day 12: 多目标检测 + NMS 去重
作者: Jeremiah
日期: 2026-08-31
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ==================== 1. 生成多物体数据 ====================
def generate_multi_data(n=2000):
    images = []
    labels = []  # 每张图 3 个框 [cx, cy, w, h]，不足补 0
    
    for _ in range(n):
        img = torch.zeros(3, 64, 64)  # 图更大，放多个物体
        num_objs = np.random.randint(2, 4)  # 2 或 3 个矩形
        boxes = []
        attempts = 0
        
        while len(boxes) < num_objs and attempts < 100:
            attempts += 1
            w = np.random.uniform(0.1, 0.3)
            h = np.random.uniform(0.1, 0.3)
            cx = np.random.uniform(w/2 + 0.05, 1 - w/2 - 0.05)
            cy = np.random.uniform(h/2 + 0.05, 1 - h/2 - 0.05)
            
            # 检查是否和已有框重叠太多
            overlap = False
            for b in boxes:
                iou = compute_iou_tensor(
                    torch.tensor([cx, cy, w, h]),
                    torch.tensor(b)
                )
                if iou > 0.1:  # 不允许重叠太多
                    overlap = True
                    break
            if not overlap:
                boxes.append([cx, cy, w, h])
        
        # 画框
        for idx, (cx, cy, w, h) in enumerate(boxes):
            color = torch.rand(3)
            px = int(cx * 64)
            py = int(cy * 64)
            pw = int(w * 64)
            ph = int(h * 64)
            x1 = max(0, px - pw // 2)
            y1 = max(0, py - ph // 2)
            x2 = min(64, px + pw // 2)
            y2 = min(64, py + ph // 2)
            img[0, y1:y2, x1:x2] = color[0]
            img[1, y1:y2, x1:x2] = color[1]
            img[2, y1:y2, x1:x2] = color[2]
        
        # 补到 3 个框（不足补 0）
        while len(boxes) < 3:
            boxes.append([0, 0, 0, 0])
        
        img = img + torch.randn(3, 64, 64) * 0.05
        img = torch.clamp(img, 0, 1)
        images.append(img)
        labels.append(torch.tensor(boxes, dtype=torch.float32))
    
    return torch.stack(images), torch.stack(labels)

def compute_iou_tensor(box1, box2):
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # 修复：用 torch.max/torch.min 而不是 Python 的 max/min
    xi1 = torch.max(b1_x1, b2_x1)
    yi1 = torch.max(b1_y1, b2_y1)
    xi2 = torch.min(b1_x2, b2_x2)
    yi2 = torch.min(b1_y2, b2_y2)
    
    inter = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else torch.tensor(0.0)

X, Y = generate_multi_data(2000)
idx = torch.randperm(2000)
X_train, Y_train = X[idx[:1600]], Y[idx[:1600]]
X_test, Y_test = X[idx[1600:]], Y[idx[1600:]]

print(f"训练集: {len(X_train)} 张")
print(f"测试集: {len(X_test)} 张")

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=50, shuffle=False)

# ==================== 2. 模型：输出 5 个候选框 ====================
class MultiYOLO(nn.Module):
    def __init__(self, num_boxes=5):
        super().__init__()
        self.num_boxes = num_boxes
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 64->32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),  # 8->4
        )
        
        # 每个框输出 5 个数：cx, cy, w, h, conf
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_boxes * 5),
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = x.view(-1, self.num_boxes, 5)
        x[:, :, :4] = torch.sigmoid(x[:, :, :4])  # 坐标 0~1
        x[:, :, 4] = torch.sigmoid(x[:, :, 4])       # 置信度 0~1
        return x

# ==================== 3. 损失函数 ====================
"""
对每张图：
- 模型输出 5 个候选框
- 真实有 3 个框（可能只有 2 个有效，第 3 个是 [0,0,0,0]）
- 对每个真实框，找预测框中 IoU 最大的那个
- 匹配上的：坐标 MSE + conf=1
- 没匹配的预测框：conf=0
"""
def detection_loss(pred, target, num_boxes=5):
    """
    pred: [batch, 5, 5]  (cx, cy, w, h, conf)
    target: [batch, 3, 4] (cx, cy, w, h)，pad 到 3 个
    """
    batch_size = pred.size(0)
    coord_loss = 0
    conf_loss = 0
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    
    for b in range(batch_size):
        p = pred[b]      # [5, 5]
        t = target[b]    # [3, 4]
        
        # 找出有效的真实框（不是 [0,0,0,0]）
        valid = (t.sum(dim=1) > 0.01)
        valid_t = t[valid]  # [N_valid, 4]
        
        if len(valid_t) == 0:
            # 没有物体，所有预测框置信度应该为 0
            conf_loss += bce(p[:, 4], torch.zeros(5, device=p.device))
            continue
        
        # 计算 IoU 矩阵 [5 预测框, N_valid 真实框]
        iou_matrix = torch.zeros(5, len(valid_t), device=p.device)
        for i in range(5):
            for j in range(len(valid_t)):
                iou_matrix[i, j] = compute_iou_tensor(p[i, :4], valid_t[j])
        
        # 每个真实框匹配最佳预测框
        matched_pred = set()
        for j in range(len(valid_t)):
            best_i = iou_matrix[:, j].argmax()
            matched_pred.add(best_i.item())
            
            # 坐标损失
            coord_loss += mse(p[best_i, :4], valid_t[j])
            # 置信度应该为 1
            conf_loss += bce(p[best_i, 4:5], torch.ones(1, device=p.device))
        
        # 没匹配的预测框，置信度应该为 0
        for i in range(5):
            if i not in matched_pred:
                conf_loss += bce(p[i, 4:5], torch.zeros(1, device=p.device))
    
    return coord_loss / batch_size + conf_loss / batch_size

# ==================== 4. 训练 ====================
model = MultiYOLO(num_boxes=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n开始训练...")
for epoch in range(1, 31):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = detection_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d} | Loss: {total_loss / len(train_loader):.4f}")

print("\n训练完成！")

# ==================== 5. NMS 函数 ====================
def nms(boxes, confidences, iou_threshold=0.5):
    """
    boxes: [N, 4] (cx, cy, w, h)
    confidences: [N]
    返回保留的索引列表
    """
    if len(boxes) == 0:
        return []
    
    # 按置信度降序
    order = torch.argsort(confidences, descending=True)
    keep = []
    
    while len(order) > 0:
        i = order[0].item()
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # 计算当前框和剩余框的 IoU
        current_box = boxes[i]
        rest_boxes = boxes[order[1:]]
        
        ious = torch.zeros(len(rest_boxes))
        for j in range(len(rest_boxes)):
            ious[j] = compute_iou_tensor(current_box, rest_boxes[j])
        
        # 保留 IoU <= 阈值的框
        mask = ious <= iou_threshold
        order = order[1:][mask]
    
    return keep

# ==================== 6. 可视化：NMS 前 vs NMS 后 ====================
model.eval()
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

with torch.no_grad():
    for idx in range(6):
        img = X_test[idx].unsqueeze(0).to(device)
        pred = model(img)[0].cpu()  # [5, 5]
        
        ax = axes[idx // 3, idx % 3]
        ax.imshow(X_test[idx].permute(1, 2, 0).numpy())
        
        # 画真实框（绿色）
        for j in range(3):
            t = Y_test[idx, j].numpy()
            if t.sum() > 0.01:  # 有效框
                tx1 = (t[0] - t[2]/2) * 64
                ty1 = (t[1] - t[3]/2) * 64
                rect = patches.Rectangle((tx1, ty1), t[2]*64, t[3]*64,
                                         linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
        
        # NMS 前：所有置信度 > 0.3 的框（黄色虚线）
        pre_boxes = []
        pre_confs = []
        for i in range(5):
            if pred[i, 4] > 0.3:
                p = pred[i].numpy()
                px1 = (p[0] - p[2]/2) * 64
                py1 = (p[1] - p[3]/2) * 64
                rect = patches.Rectangle((px1, py1), p[2]*64, p[3]*64,
                                         linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                pre_boxes.append(pred[i, :4])
                pre_confs.append(pred[i, 4])
        
        # NMS 后：保留的框（红色实线）
        if len(pre_boxes) > 0:
            keep = nms(torch.stack(pre_boxes), torch.stack(pre_confs), iou_threshold=0.5)
            for k in keep:
                p = pred[k].numpy()
                px1 = (p[0] - p[2]/2) * 64
                py1 = (p[1] - p[3]/2) * 64
                rect = patches.Rectangle((px1, py1), p[2]*64, p[3]*64,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
        
        n_pre = len(pre_boxes)
        n_post = len(keep) if len(pre_boxes) > 0 else 0
        ax.set_title(f"NMS前:{n_pre}框 → NMS后:{n_post}框", fontsize=9)
        ax.axis('off')

plt.suptitle("Green=Truth | Yellow dashed=Before NMS | Red=After NMS", fontsize=14)
plt.tight_layout()
plt.savefig('nms_demo.png', dpi=150)
plt.close()
print("图已保存: nms_demo.png")

print("\n今日任务完成！")
print("1. 观察：NMS 前黄色虚线框是不是很多重叠？")
print("2. 观察：NMS 后红色框是不是和绿色真实框基本对应？")
print("3. 思考：如果 IoU 阈值设成 0.9，会删掉更多还是更少框？")
print("   答案：阈值越高，越不容易删，保留越多框（可能重复）")
print("4. Obsidian: [[NMS]] [[多目标检测]]")