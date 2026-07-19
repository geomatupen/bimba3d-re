# Metric-Specific Best and Lowest Performing 5000-Step Previews

These folders contain one best and one lowest-performing project for each final metric improvement from the compact Ridge test run.

Model:
`model_compact-ridge_20260712_131550_final_offline_data_june_27-training-data-compact-ridge`

For PSNR and SSIM, higher final metric difference is better. For LPIPS, improvement is computed as `baseline LPIPS - model LPIPS`, so a higher value is also better.

| Metric | Folder | Project | Delta |
| --- | --- | --- | --- |
| PSNR best | `psnr_best_pix4d_forensic` | Pix4d_forensic | `+1.000858` |
| PSNR lowest | `psnr_lowest_4_thomas_more_church` | 4-Thomas-More-Church | `-0.167141` |
| SSIM best | `ssim_best_pix4d_forensic` | Pix4d_forensic | `+0.148877` |
| SSIM lowest | `ssim_lowest_12_maisonneuve_market` | 12-Maisonneuve Market | `-0.001925` |
| LPIPS best | `lpips_best_pix4d_forensic` | Pix4d_forensic | `+0.232032` |
| LPIPS lowest improvement | `lpips_lowest_12_maisonneuve_market` | 12-Maisonneuve Market | `+0.001848` |

Inside each folder:

- `baseline_preview_005000.png`
- `model_compact_ridge_preview_005000.png`
