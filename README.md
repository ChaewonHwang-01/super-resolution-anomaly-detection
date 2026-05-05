# Super-Resolution Based Anomaly Detection

This project compares reconstruction-based anomaly detection methods on the Grid Dataset.

## Methods

- AE only
- SR+AE
- SR+VAE

## Result

| Method | AUROC |
|---|---:|
| AE only | 0.6424 |
| SR+AE | 0.5572 |
| SR+VAE | 0.9658 |

SR+VAE achieved the highest AUROC score in this experiment.

## Demo

Run the SR+VAE Gradio demo:

```bash
python grid_demo_app.py