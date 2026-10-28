| Model | Size bins | Steps (displayed MAP trajectory) | Max CDF change: time / size / domain | Max material error (g/L) |
|:--|--:|--:|:--|--:|
| cnt | 400 | 832 | 5.5e-06 / 0.0048 / 0 | 1e-12 |
| cnt_grd | 200 | 1368 | 1.4e-07 / 0.0013 / 0 | 2.6e-12 |

The refinements above hold parameters fixed. A separate calibration-only refit on twice as many bins gives:

| Model | Max calibration CDF change after refit | Max concentration change (g/L) |
|:--|--:|--:|
| cnt | 0.0115 | 0.11 |
| cnt_grd | 0.000732 | 0.00258 |
