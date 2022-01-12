# Adversarial Attack
## File
- `eval_attack` 如果已有訓練好的模型，可直接使用此檔案做模型攻擊的評估
- `test.py` 實測test資料夾中的所有圖片(Cifar10)經過攻擊後的預測結果
    - model_08.pt download (https://ncku365-my.sharepoint.com/:u:/g/personal/p76104794_ncku_edu_tw/ESDKN9vRfupLq3wKV-tt-0UBJ6D-5uWG_MficWL5NIuvRw?e=r5rU47)
- `simba.py` 單張圖片simple black box攻擊
- `single_img_attack.py` 單張圖片各種攻擊，模型採用Train CIFAR10 with PyTorch(https://github.com/kuangliu/pytorch-cifar)進行訓練

## Requirement
cleverhans 0.0.8
blackbox-adversarial-toolbox 4.0.0