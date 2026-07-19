# Compact MLP 5000-Step Preview Images

These folders contain baseline and compact MLP `preview_005000.png` image pairs.

Model:
`model_compact-mlp_20260718_194410_compact_mlp_july18_final_offline_data_june_27-training-data-compact-featurewise-mlp`

The current test rows contain 11 completed compact MLP rows for this model. `Brasov` is not present for this specific compact MLP model in the current pipeline rows.

## Overall Best and Lowest Performing

Overall ranking uses the combined rank across final PSNR, SSIM, and LPIPS improvements. For LPIPS, improvement is computed as `baseline LPIPS - model LPIPS`.

| Folder | Project | Delta PSNR | Delta SSIM | Delta LPIPS |
| --- | --- | ---: | ---: | ---: |
| `overall_01_best_pix4d_forensic` | Pix4d_forensic | `+1.180725` | `+0.145993` | `+0.243408` |
| `overall_02_best_chatteau_circle_60_and_45_degrees` | Chatteau_circle_60_and_45_degrees | `+0.571493` | `+0.099058` | `+0.215354` |
| `overall_01_lowest_12_maisonneuve_market` | 12-Maisonneuve Market | `-0.651958` | `-0.006597` | `-0.008134` |
| `overall_02_lowest_tiburon_angel_islands_state_park_split_group_3` | Tiburon_Angel_Islands_State_Park_Split_group_3 | `-0.740726` | `+0.003910` | `+0.046354` |

## Metric-Specific Best and Lowest Performing

| Metric | Folder | Project | Delta |
| --- | --- | --- | ---: |
| PSNR best | `metric_specific_best_lowest/psnr_best_morice` | morice | `+1.302168` |
| PSNR lowest | `metric_specific_best_lowest/psnr_lowest_tiburon_angel_islands_state_park_split_group_3` | Tiburon_Angel_Islands_State_Park_Split_group_3 | `-0.740726` |
| SSIM best | `metric_specific_best_lowest/ssim_best_pix4d_forensic` | Pix4d_forensic | `+0.145993` |
| SSIM lowest | `metric_specific_best_lowest/ssim_lowest_changu` | Changu | `-0.012749` |
| LPIPS best | `metric_specific_best_lowest/lpips_best_pix4d_forensic` | Pix4d_forensic | `+0.243408` |
| LPIPS lowest | `metric_specific_best_lowest/lpips_lowest_12_maisonneuve_market` | 12-Maisonneuve Market | `-0.008134` |

Inside each folder:

- `baseline_preview_005000.png`
- `model_compact_mlp_preview_005000.png`
