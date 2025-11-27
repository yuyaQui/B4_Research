import torch.nn as nn

# モデル定義
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 子1: 特徴抽出パート（中にさらにレイヤーがいっぱいある）
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 子2: 分類パート
        self.classifier = nn.Linear(64, 10)

model = MyModel()

# .children() を実行！
children = list(model.children())
print(children)
print(len(children)) 
# 結果: 2 （self.features と self.classifier の2つだけ）