# Best and Lowest Performing 5000-Step Preview Images

This folder collects baseline/model preview pairs for selected compact Ridge and compact MLP test runs. Each selected run uses the `preview_005000.png` image from the Gaussian Splatting output.

Metric differences are final test-run values compared with the matching baseline run for the same project:

- Delta PSNR = model PSNR - baseline PSNR
- Delta SSIM = model SSIM - baseline SSIM
- Delta LPIPS = baseline LPIPS - model LPIPS

Higher delta is better for all three columns.

## Folder Structure

| Folder | Contents |
| --- | --- |
| `compact_ridge` | Overall and metric-specific best/lowest previews for the compact Ridge model |
| `compact_mlp` | Overall and metric-specific best/lowest previews for the compact MLP model |

Inside each selected project folder:

- `baseline_preview_005000.png`
- `model_compact_ridge_preview_005000.png` or `model_compact_mlp_preview_005000.png`

## Compact Ridge

Model:
`model_compact-ridge_20260712_131550_final_offline_data_june_27-training-data-compact-ridge`

Rows used: 12 completed compact Ridge test rows.

### Overall Best and Lowest Performing

Overall ranking uses the combined rank across Delta PSNR, Delta SSIM, and Delta LPIPS.

| Case | Folder | Project | Delta PSNR | Delta SSIM | Delta LPIPS |
| --- | --- | --- | ---: | ---: | ---: |
| Best 1 | `compact_ridge/01_best_pix4d_forensic` | Pix4d_forensic | `+1.000858` | `+0.148877` | `+0.232032` |
| Best 2 | `compact_ridge/02_best_morice` | morice | `+0.857925` | `+0.057201` | `+0.104905` |
| Lowest 1 | `compact_ridge/03_lowest_12_maisonneuve_market` | 12-Maisonneuve Market | `+0.003895` | `-0.001925` | `+0.001848` |
| Lowest 2 | `compact_ridge/04_lowest_4_thomas_more_church` | 4-Thomas-More-Church | `-0.167141` | `+0.005073` | `+0.007576` |

### Metric-Specific Best and Lowest Performing

| Metric | Case | Folder | Project | Delta |
| --- | --- | --- | --- | ---: |
| PSNR | Best | `compact_ridge/metric_specific_best_lowest/psnr_best_pix4d_forensic` | Pix4d_forensic | `+1.000858` |
| PSNR | Lowest | `compact_ridge/metric_specific_best_lowest/psnr_lowest_4_thomas_more_church` | 4-Thomas-More-Church | `-0.167141` |
| SSIM | Best | `compact_ridge/metric_specific_best_lowest/ssim_best_pix4d_forensic` | Pix4d_forensic | `+0.148877` |
| SSIM | Lowest | `compact_ridge/metric_specific_best_lowest/ssim_lowest_12_maisonneuve_market` | 12-Maisonneuve Market | `-0.001925` |
| LPIPS | Best | `compact_ridge/metric_specific_best_lowest/lpips_best_pix4d_forensic` | Pix4d_forensic | `+0.232032` |
| LPIPS | Lowest improvement | `compact_ridge/metric_specific_best_lowest/lpips_lowest_12_maisonneuve_market` | 12-Maisonneuve Market | `+0.001848` |

## Compact MLP

Model:
`model_compact-mlp_20260718_194410_compact_mlp_july18_final_offline_data_june_27-training-data-compact-featurewise-mlp`

Rows used: 11 completed compact MLP test rows. `Brasov` is not present for this specific compact MLP model in the current test rows.

### Overall Best and Lowest Performing

Overall ranking uses the combined rank across Delta PSNR, Delta SSIM, and Delta LPIPS.

| Case | Folder | Project | Delta PSNR | Delta SSIM | Delta LPIPS |
| --- | --- | --- | ---: | ---: | ---: |
| Best 1 | `compact_mlp/overall_01_best_pix4d_forensic` | Pix4d_forensic | `+1.180725` | `+0.145993` | `+0.243408` |
| Best 2 | `compact_mlp/overall_02_best_chatteau_circle_60_and_45_degrees` | Chatteau_circle_60_and_45_degrees | `+0.571493` | `+0.099058` | `+0.215354` |
| Lowest 1 | `compact_mlp/overall_01_lowest_12_maisonneuve_market` | 12-Maisonneuve Market | `-0.651958` | `-0.006597` | `-0.008134` |
| Lowest 2 | `compact_mlp/overall_02_lowest_tiburon_angel_islands_state_park_split_group_3` | Tiburon_Angel_Islands_State_Park_Split_group_3 | `-0.740726` | `+0.003910` | `+0.046354` |

### Metric-Specific Best and Lowest Performing

| Metric | Case | Folder | Project | Delta |
| --- | --- | --- | --- | ---: |
| PSNR | Best | `compact_mlp/metric_specific_best_lowest/psnr_best_morice` | morice | `+1.302168` |
| PSNR | Lowest | `compact_mlp/metric_specific_best_lowest/psnr_lowest_tiburon_angel_islands_state_park_split_group_3` | Tiburon_Angel_Islands_State_Park_Split_group_3 | `-0.740726` |
| SSIM | Best | `compact_mlp/metric_specific_best_lowest/ssim_best_pix4d_forensic` | Pix4d_forensic | `+0.145993` |
| SSIM | Lowest | `compact_mlp/metric_specific_best_lowest/ssim_lowest_changu` | Changu | `-0.012749` |
| LPIPS | Best | `compact_mlp/metric_specific_best_lowest/lpips_best_pix4d_forensic` | Pix4d_forensic | `+0.243408` |
| LPIPS | Lowest | `compact_mlp/metric_specific_best_lowest/lpips_lowest_12_maisonneuve_market` | 12-Maisonneuve Market | `-0.008134` |
