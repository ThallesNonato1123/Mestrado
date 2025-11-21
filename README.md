# Autoencoder Fashion-MNIST

Este projeto implementa um **autoencoder para reconstrução de imagens do Fashion-MNIST** usando PyTorch, com suporte a **K-Fold Cross-Validation** e treinamento do modelo final em todo o dataset.  

O projeto calcula métricas de reconstrução como **MSE**, **SSIM**, e é possível adicionar MAE, RMSE e PSNR. Também salva plots e imagens reconstruídas na pasta `outputs/`.  

## 🚀 Como rodar

Execute o script principal com os argumentos desejados:

```bash
python main.py --k 5 --epochs 50 --batch-size 64 --lr 0.001

| Argumento      | Tipo   | Default | Descrição                                |
|----------------|--------|---------|------------------------------------------|
| `--k`          | int    | 5       | Número de folds para K-Fold Cross-Validation |
| `--epochs`     | int    | 50      | Número de épocas de treinamento          |
| `--batch-size` | int    | 64      | Tamanho do batch para DataLoader         |
| `--lr`         | float  | 0.001   | Learning rate do otimizador Adam         |
