# repdata tables (generated)

References (1x, 3 seeds, seed 0): TCN 0.22M = o_ls05, TCN 19.9M = o_ls05_C640, MS-TCN 2.6M = m5c_C192ls, MS-TCN 19.7M = m5c_C528ls

## A. Repetition sweep (gradient steps x multiplier; passes over the 3.87M stored steps / 1.29M unique)

| model | setting | final train loss (logged; it/ep) | NLL train | NLL held-out | gap | PFF f0 fine3 | vs ref | SKC fine3 | vs ref | Metrica dev | vs ref (2 games) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rd_tcn02_x025 | TCN 0.22M x0.25: 2.1 passes stored / 6.3 unique | 0.3362 (1000) | 0.1762 | 0.1929 | 0.0167 | 0.7375 | -0.0528 ± 0.0072 (0/12) | 0.5753 | -0.0097 ± 0.0033 (6/20) | 0.7151 | -0.0166 (0/2) |
| rd_tcn02_x05 | TCN 0.22M x0.5: 4.2 passes stored / 12.7 unique | 0.3042 (2000) | 0.1536 | 0.1766 | 0.0230 | 0.7702 | -0.0201 ± 0.0044 (1/12) | 0.5769 | -0.0081 ± 0.0026 (4/20) | 0.7216 | -0.0100 (0/2) |
| o_ls05 | TCN 0.22M x1: 8.5 passes stored / 25.4 unique | 0.2720 (4000) | 0.1318 | 0.1621 | 0.0303 | 0.7903 | +0.0000 ± 0.0000 (0/12) | 0.5850 | +0.0000 ± 0.0000 (0/20) | 0.7316 | +0.0000 (0/2) |
| rd_tcn02_x2 | TCN 0.22M x2: 16.9 passes stored / 50.8 unique | 0.2615 (8000) | 0.1113 | 0.1616 | 0.0503 | 0.7921 | +0.0018 ± 0.0025 (6/12) | 0.5848 | -0.0002 ± 0.0024 (10/20) | 0.7351 | +0.0035 (1/2) |
| rd_tcn02_x4 | TCN 0.22M x4: 33.8 passes stored / 101.5 unique | 0.2482 (16000) | 0.0898 | 0.1691 | 0.0793 | 0.7905 | +0.0003 ± 0.0035 (8/12) | 0.5744 | -0.0106 ± 0.0034 (5/20) | 0.7286 | -0.0030 (1/2) |
| rd_tcn20_x025 | TCN 19.9M x0.25: 2.1 passes stored / 6.3 unique | 0.3099 (1000) | 0.1492 | 0.1725 | 0.0232 | 0.7732 | -0.0219 ± 0.0047 (1/12) | 0.5846 | +0.0067 ± 0.0032 (13/20) | 0.7285 | -0.0132 (0/2) |
| rd_tcn20_x05 | TCN 19.9M x0.5: 4.2 passes stored / 12.7 unique | 0.2706 (2000) | 0.1213 | 0.1571 | 0.0358 | 0.7939 | -0.0012 ± 0.0037 (4/12) | 0.5944 | +0.0165 ± 0.0032 (17/20) | 0.7338 | -0.0079 (0/2) |
| o_ls05_C640 | TCN 19.9M x1: 8.5 passes stored / 25.4 unique | 0.2324 (4000) | 0.0923 | 0.1585 | 0.0662 | 0.7951 | +0.0000 ± 0.0000 (0/12) | 0.5779 | +0.0000 ± 0.0000 (0/20) | 0.7416 | +0.0000 (0/2) |
| rd_tcn20_x2 | TCN 19.9M x2: 16.9 passes stored / 50.8 unique | 0.2101 (8000) | 0.0650 | 0.1743 | 0.1094 | 0.7936 | -0.0014 ± 0.0030 (4/12) | 0.5731 | -0.0048 ± 0.0042 (7/20) | 0.7320 | -0.0096 (0/2) |
| rd_tcn20_x4 | TCN 19.9M x4: 33.8 passes stored / 101.5 unique | 0.1960 (15750) | - | - | - | - | - | - | - | - | - |
| rd_ms26_x025 | MS-TCN 2.6M x0.25: 3.2 passes stored / 9.5 unique | 1.1365 (9) | 0.1623 | 0.1855 | 0.0232 | 0.7756 | -0.0189 ± 0.0043 (1/12) | 0.5483 | -0.0075 ± 0.0039 (7/20) | 0.7020 | +0.0073 (1/2) |
| rd_ms26_x05 | MS-TCN 2.6M x0.5: 6.3 passes stored / 19.0 unique | 1.0582 (19) | 0.1377 | 0.1738 | 0.0360 | 0.7947 | +0.0003 ± 0.0048 (6/12) | 0.5690 | +0.0132 ± 0.0036 (16/20) | 0.7096 | +0.0150 (2/2) |
| m5c_C192ls | MS-TCN 2.6M x1: 12.7 passes stored / 38.1 unique | 1.0138 (39) | 0.1100 | 0.1729 | 0.0629 | 0.7944 | +0.0000 ± 0.0000 (0/12) | 0.5558 | +0.0000 ± 0.0000 (0/20) | 0.6947 | +0.0000 (0/2) |
| rd_ms26_x2 | MS-TCN 2.6M x2: 25.4 passes stored / 76.1 unique | 0.8541 (79) | 0.0841 | 0.1902 | 0.1061 | 0.8003 | +0.0059 ± 0.0033 (9/12) | 0.5511 | -0.0047 ± 0.0034 (9/20) | 0.6896 | -0.0051 (0/2) |
| rd_ms26_x4 | MS-TCN 2.6M x4: 50.8 passes stored / 152.3 unique | 1.0197 (31) | - | - | - | - | - | - | - | - | - |
| rd_ms20_x025 | MS-TCN 19.7M x0.25: 3.2 passes stored / 9.5 unique | 1.0839 (9) | 0.1503 | 0.1787 | 0.0284 | 0.7891 | -0.0130 ± 0.0047 (4/12) | 0.5586 | +0.0029 ± 0.0045 (9/20) | 0.7035 | +0.0173 (2/2) |
| rd_ms20_x05 | MS-TCN 19.7M x0.5: 6.3 passes stored / 19.0 unique | 1.0092 (19) | 0.1255 | 0.1719 | 0.0464 | 0.7976 | -0.0045 ± 0.0036 (4/12) | 0.5556 | +0.0000 ± 0.0038 (9/20) | 0.7033 | +0.0171 (2/2) |
| m5c_C528ls | MS-TCN 19.7M x1: 12.7 passes stored / 38.1 unique | 0.9341 (39) | 0.0974 | 0.1797 | 0.0823 | 0.8021 | +0.0000 ± 0.0000 (0/12) | 0.5556 | +0.0000 ± 0.0000 (0/20) | 0.6862 | +0.0000 (0/2) |
| rd_ms20_x2 | MS-TCN 19.7M x2: 25.4 passes stored / 76.1 unique | 0.8039 (75) | - | - | - | - | - | - | - | - | - |
| rd_ms20_x4 | MS-TCN 19.7M x4: 50.8 passes stored / 152.3 unique | 0.8436 (54) | - | - | - | - | - | - | - | - | - |

## E. Seed noise (1x recipe)

| model | setting | final train loss (logged; it/ep) | NLL train | NLL held-out | gap | PFF f0 fine3 | vs ref | SKC fine3 | vs ref | Metrica dev | vs ref (2 games) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rd_tcn02_s0cuda | TCN 0.22M seed s0cuda | - | - | - | - | - | - | - | - | - | - |
| rd_tcn02_s1 | TCN 0.22M seed s1 | - | - | - | - | - | - | - | - | - | - |
| rd_tcn02_s2 | TCN 0.22M seed s2 | - | - | - | - | - | - | - | - | - | - |
| rd_tcn20_s1 | TCN 19.9M seed s1 | - | - | - | - | - | - | - | - | - | - |
| rd_tcn20_s2 | TCN 19.9M seed s2 | - | - | - | - | - | - | - | - | - | - |
| rd_ms26_s1 | MS-TCN 2.6M seed s1 | - | - | - | - | - | - | - | - | - | - |
| rd_ms26_s2 | MS-TCN 2.6M seed s2 | - | - | - | - | - | - | - | - | - | - |
| rd_ms20_s1 | MS-TCN 19.7M seed s1 | - | - | - | - | - | - | - | - | - | - |
| rd_ms20_s2 | MS-TCN 19.7M seed s2 | - | - | - | - | - | - | - | - | - | - |

Between-seed SD of the bench means (seeds 0 / 1 / 2; o_ls05 seed 0 = MPS; with its CUDA seed-0 rerun in brackets) vs the mean paired game-level SE of the A-sweep diffs of that size:

| size | bench | seed means | SD (n-1) | typical paired SE (A rows) |
|---|---|---|---|---|
| TCN 0.22M | PFF | - | - | - |
| TCN 0.22M | SKC | - | - | - |
| TCN 0.22M | Metrica | - | - | - |
| TCN 19.9M | PFF | - | - | - |
| TCN 19.9M | SKC | - | - | - |
| TCN 19.9M | Metrica | - | - | - |
| MS-TCN 2.6M | PFF | - | - | - |
| MS-TCN 2.6M | SKC | - | - | - |
| MS-TCN 2.6M | Metrica | - | - | - |
| MS-TCN 19.7M | PFF | - | - | - |
| MS-TCN 19.7M | SKC | - | - | - |
| MS-TCN 19.7M | Metrica | - | - | - |

20M vs shipped size, seed-averaged (mean of 3 seeds each; SE = sqrt(SD20^2/3 + SDs^2/3)) next to the seed-0 paired diff ± game-level SE:

| family | bench | seed-avg diff ± seed SE | seed-0 paired diff (game SE) | per-seed paired diffs (seed k vs seed k) |
|---|---|---|---|---|
| TCN | PFF | - | - | - |
| TCN | SKC | - | - | - |
| TCN | Metrica | - | - | - |
| MS-TCN | PFF | - | - | - |
| MS-TCN | SKC | - | - | - |
| MS-TCN | Metrica | - | - | - |

## E / A on the Mac GPU (MPS; seeds of one size share one device; o_ls05 seed 0 is MPS)

| model | setting | final train loss (logged; it/ep) | NLL train | NLL held-out | gap | PFF f0 fine3 | vs ref | SKC fine3 | vs ref | Metrica dev | vs ref (2 games) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rd_tcn02_s1_mps | TCN 0.22M MPS, vs MPS seed 0 o_ls05 | 0.2796 (4000) | - | - | - | 0.7871 | -0.0032 ± 0.0031 (3/12) | 0.5831 | -0.0020 ± 0.0027 (9/20) | 0.7278 | -0.0038 (0/2) |
| rd_tcn02_s2_mps | TCN 0.22M MPS, vs MPS seed 0 o_ls05 | 0.2790 (4000) | - | - | - | 0.7866 | -0.0037 ± 0.0028 (5/12) | 0.5882 | +0.0032 ± 0.0027 (13/20) | 0.7342 | +0.0026 (1/2) |
| rd_tcn20_s0_mps | TCN 19.9M MPS, vs MPS seed 0 rd_tcn20_s0_mps | 0.2312 (4000) | - | - | - | 0.8010 | +0.0000 ± 0.0000 (0/12) | 0.5787 | +0.0000 ± 0.0000 (0/20) | 0.7363 | +0.0000 (0/2) |
| rd_tcn20_s1_mps | TCN 19.9M MPS, vs MPS seed 0 rd_tcn20_s0_mps | 0.2390 (4000) | - | - | - | 0.7946 | -0.0063 ± 0.0021 (3/12) | 0.5816 | +0.0029 ± 0.0027 (9/20) | 0.7272 | -0.0091 (0/2) |
| rd_tcn20_s2_mps | TCN 19.9M MPS, vs MPS seed 0 rd_tcn20_s0_mps | 0.2377 (4000) | - | - | - | 0.7971 | -0.0038 ± 0.0034 (5/12) | 0.5841 | +0.0053 ± 0.0026 (15/20) | 0.7367 | +0.0003 (1/2) |
| rd_tcn20_x4_mps | TCN 19.9M x4_mps vs MPS seed 0 | 0.2004 (16000) | - | - | - | - | - | - | - | - | - |
| rd_tcn20_x4_mps | TCN 19.9M x4_mps vs CUDA seed 0 o_ls05_C640 | 0.2004 (16000) | - | - | - | - | - | - | - | - | - |
| rd_ms26_s0_mps | MS-TCN 2.6M MPS, vs MPS seed 0 rd_ms26_s0_mps | 1.0068 (39) | - | - | - | 0.7975 | +0.0000 ± 0.0000 (0/12) | 0.5521 | +0.0000 ± 0.0000 (0/20) | 0.6963 | +0.0000 (0/2) |
| rd_ms26_s1_mps | MS-TCN 2.6M MPS, vs MPS seed 0 rd_ms26_s0_mps | 0.9318 (39) | - | - | - | 0.8022 | +0.0046 ± 0.0029 (8/12) | 0.5632 | +0.0111 ± 0.0036 (14/20) | 0.7058 | +0.0096 (2/2) |
| rd_ms26_s2_mps | MS-TCN 2.6M MPS, vs MPS seed 0 rd_ms26_s0_mps | 0.9440 (39) | - | - | - | 0.7968 | -0.0007 ± 0.0036 (3/12) | 0.5650 | +0.0129 ± 0.0037 (14/20) | 0.7093 | +0.0131 (2/2) |
| rd_ms26_x4_mps | MS-TCN 2.6M x4_mps vs MPS seed 0 | 0.8215 (159) | - | - | - | - | - | - | - | - | - |
| rd_ms26_x4_mps | MS-TCN 2.6M x4_mps vs CUDA seed 0 m5c_C192ls | 0.8215 (159) | - | - | - | - | - | - | - | - | - |
| rd_ms20_s0_mps | MS-TCN 19.7M MPS, vs MPS seed 0 rd_ms20_s0_mps | 0.9400 (39) | - | - | - | 0.8004 | +0.0000 ± 0.0000 (0/12) | - | - | - | - |
| rd_ms20_s1_mps | MS-TCN 19.7M MPS, vs MPS seed 0 rd_ms20_s0_mps | 0.8920 (39) | - | - | - | - | - | - | - | - | - |
| rd_ms20_s2_mps | MS-TCN 19.7M MPS, vs MPS seed 0 rd_ms20_s0_mps | 0.8944 (39) | - | - | - | - | - | - | - | - | - |
| rd_ms20_x2_mps | MS-TCN 19.7M x2_mps vs MPS seed 0 | 0.7982 (79) | - | - | - | - | - | - | - | - | - |
| rd_ms20_x2_mps | MS-TCN 19.7M x2_mps vs CUDA seed 0 m5c_C528ls | 0.7982 (79) | - | - | - | - | - | - | - | - | - |
| rd_ms20_x4_mps | MS-TCN 19.7M x4_mps vs MPS seed 0 | 0.7635 (89) | - | - | - | - | - | - | - | - | - |
| rd_ms20_x4_mps | MS-TCN 19.7M x4_mps vs CUDA seed 0 m5c_C528ls | 0.7635 (89) | - | - | - | - | - | - | - | - | - |

MPS seed noise: SD (n-1) of the bench means over seeds 0 / 1 / 2 (all MPS):

| size | PFF means | PFF SD | SKC means | SKC SD | Metrica means | Metrica SD |
|---|---|---|---|---|---|---|
| TCN 0.22M | 0.7903 / 0.7871 / 0.7866 | 0.0020 | 0.5850 / 0.5831 / 0.5882 | 0.0026 | 0.7316 / 0.7278 / 0.7342 | 0.0033 |
| TCN 19.9M | 0.8010 / 0.7946 / 0.7971 | 0.0032 | 0.5787 / 0.5816 / 0.5841 | 0.0027 | 0.7363 / 0.7272 / 0.7367 | 0.0054 |
| MS-TCN 2.6M | 0.7975 / 0.8022 / 0.7968 | 0.0029 | 0.5521 / 0.5632 / 0.5650 | 0.0070 | 0.6963 / 0.7058 / 0.7093 | 0.0068 |
| MS-TCN 19.7M | - | - | - | - | - | - |

## Decisive: x0.5 vs x1, both MPS, 3 seeds each (prereg ADDITION 16:22)

| size | bench | x0.5 seed means | x0.5 SD | x1 seed means | x1 SD | diff of 3-seed means ± seed SE | diff / seed SE | 9 cross-seed paired diffs: mean of means, range, wins of the 9 means |
|---|---|---|---|---|---|---|---|---|
| MS-TCN 2.6M | PFF | 0.7979 / 0.7915 / 0.7949 | 0.0032 | 0.7975 / 0.8022 / 0.7968 | 0.0029 | -0.0041 ± 0.0025 | -1.6 | -0.0041, [-0.0107, +0.0011], 2/9 |
| MS-TCN 2.6M | SKC | 0.5691 / 0.5653 / 0.5621 | 0.0035 | 0.5521 / 0.5632 / 0.5650 | 0.0070 | +0.0054 ± 0.0045 | +1.2 | +0.0054, [-0.0029, +0.0169], 7/9 |
| MS-TCN 2.6M | Metrica | 0.7085 / 0.7138 / 0.7160 | 0.0039 | 0.6963 / 0.7058 / 0.7093 | 0.0068 | +0.0090 ± 0.0045 | +2.0 | +0.0090, [-0.0008, +0.0197], 8/9 |
| TCN 19.9M | PFF | 0.7970 / 0.7887 / 0.7944 | 0.0042 | 0.8010 / 0.7946 / 0.7971 | 0.0032 | -0.0042 ± 0.0031 | -1.4 | -0.0042, [-0.0122, +0.0023], 1/9 |
| TCN 19.9M | SKC | 0.5868 / 0.5884 / 0.5890 | 0.0011 | 0.5787 / 0.5816 / 0.5841 | 0.0027 | +0.0066 ± 0.0017 | +3.9 | +0.0066, [+0.0027, +0.0103], 9/9 |
| TCN 19.9M | Metrica | 0.7337 / 0.7399 / 0.7310 | 0.0046 | 0.7363 / 0.7272 / 0.7367 | 0.0054 | +0.0015 ± 0.0041 | +0.4 | +0.0015, [-0.0057, +0.0127], 5/9 |

## D. Data-size scan (same gradient steps as 1x -> x4 / x2 passes per game at 25 % / 50 %)

| model | setting | final train loss (logged; it/ep) | NLL train | NLL held-out | gap | PFF f0 fine3 | vs ref | SKC fine3 | vs ref | Metrica dev | vs ref (2 games) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rd_tcn02_g25 | TCN 0.22M 11 games (9 PFF + 2 DFL) | - | - | - | - | - | - | - | - | - | - |
| rd_tcn02_g50 | TCN 0.22M 22 games (18 PFF + 4 DFL) | - | - | - | - | - | - | - | - | - | - |
| o_ls05 | TCN 0.22M 43 games | 0.2720 (4000) | 0.1318 | 0.1621 | 0.0303 | 0.7903 | +0.0000 ± 0.0000 (0/12) | 0.5850 | +0.0000 ± 0.0000 (0/20) | 0.7316 | +0.0000 (0/2) |
| rd_tcn20_g25 | TCN 19.9M 11 games (9 PFF + 2 DFL) | - | - | - | - | - | - | - | - | - | - |
| rd_tcn20_g50 | TCN 19.9M 22 games (18 PFF + 4 DFL) | - | - | - | - | - | - | - | - | - | - |
| o_ls05_C640 | TCN 19.9M 43 games | 0.2324 (4000) | 0.0923 | 0.1585 | 0.0662 | 0.7951 | +0.0000 ± 0.0000 (0/12) | 0.5779 | +0.0000 ± 0.0000 (0/20) | 0.7416 | +0.0000 (0/2) |
| rd_ms26_g25 | MS-TCN 2.6M 11 games (9 PFF + 2 DFL) | - | - | - | - | - | - | - | - | - | - |
| rd_ms26_g50 | MS-TCN 2.6M 22 games (18 PFF + 4 DFL) | - | - | - | - | - | - | - | - | - | - |
| m5c_C192ls | MS-TCN 2.6M 43 games | 1.0138 (39) | 0.1100 | 0.1729 | 0.0629 | 0.7944 | +0.0000 ± 0.0000 (0/12) | 0.5558 | +0.0000 ± 0.0000 (0/20) | 0.6947 | +0.0000 (0/2) |
| rd_ms20_g25 | MS-TCN 19.7M 11 games (9 PFF + 2 DFL) | - | - | - | - | - | - | - | - | - | - |
| rd_ms20_g50 | MS-TCN 19.7M 22 games (18 PFF + 4 DFL) | - | - | - | - | - | - | - | - | - | - |
| m5c_C528ls | MS-TCN 19.7M 43 games | 0.9341 (39) | 0.0974 | 0.1797 | 0.0823 | 0.8021 | +0.0000 ± 0.0000 (0/12) | 0.5556 | +0.0000 ± 0.0000 (0/20) | 0.6862 | +0.0000 (0/2) |

Rough power-law slopes (least squares on log-log over 11 / 22 / 43 games; held-out NLL and (1 - score)):

| size | held-out NLL slope | PFF (1-fine3) slope | SKC (1-fine3) slope | NLL at 11 / 22 / 43 |
|---|---|---|---|---|
| TCN 0.22M | - | - | - | - / - / 0.1621 |
| TCN 19.9M | - | - | - | - / - / 0.1585 |
| MS-TCN 2.6M | - | - | - | - / - / 0.1729 |
| MS-TCN 19.7M | - | - | - | - / - / 0.1797 |

## B. Less exact repetition at equal compute (1x steps, 12 stored degradation seeds)

| model | setting | final train loss (logged; it/ep) | NLL train | NLL held-out | gap | PFF f0 fine3 | vs ref | SKC fine3 | vs ref | Metrica dev | vs ref (2 games) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rd_tcn02_sd12 | TCN 0.22M 12 seeds | - | - | - | - | - | - | - | - | - | - |
| rd_tcn20_sd12 | TCN 19.9M 12 seeds | - | - | - | - | - | - | - | - | - | - |
| rd_ms26_sd12 | MS-TCN 2.6M 12 seeds | - | - | - | - | - | - | - | - | - | - |
| rd_ms20_sd12 | MS-TCN 19.7M 12 seeds | - | - | - | - | - | - | - | - | - | - |
| o_ls05_s8 | context: TCN 0.22M, 8 seeds (round 4, existing) | 0.2826 (4000) | 0.1341 | 0.1609 | 0.0269 | 0.7898 | -0.0005 ± 0.0021 (5/12) | - | - | - | - |
| r5_otf_tL | context: TCN 0.22M, fresh degradation per crop (R5-6b, existing) | 0.2855 (4000) | 0.1377 | 0.1629 | 0.0251 | 0.7851 | -0.0052 ± 0.0032 (5/12) | - | - | - | - |

## C. Memorisation diagnostic, size ladders (NLL of the true label, stored degraded inputs, seeds 41-43)

| family | model | parameters | final train loss (logged) | NLL train (12 PFF training games) | NLL held-out (12 fold-0 games) | gap | ratio held/train | acc train | acc held |
|---|---|---|---|---|---|---|---|---|---|
| TCN | o_ls05 | 215,812 | 0.2720 | 0.1318 | 0.1621 | +0.0303 | 1.23 | 0.9608 | 0.9483 |
| TCN | o_ls05_C208 | 2,139,076 | 0.2497 | 0.1070 | 0.1587 | +0.0517 | 1.48 | 0.9713 | 0.9508 |
| TCN | o_ls05_C640 | 19,852,804 | 0.2324 | 0.0923 | 0.1585 | +0.0662 | 1.72 | 0.9774 | 0.9518 |
| TCN | o_ls05_C2032 | 198,802,756 | 0.2283 | 0.0871 | 0.1599 | +0.0728 | 1.84 | 0.9795 | 0.9515 |
| TCN | o_ls05_C3232 | 502,369,156 | 0.2305 | 0.0878 | 0.1597 | +0.0718 | 1.82 | 0.9789 | 0.9513 |
| MS-TCN | m5c_C192ls | 2,630,317 | 1.0138 | 0.1100 | 0.1729 | +0.0629 | 1.57 | 0.9695 | 0.9480 |
| MS-TCN | m5c_C528ls | 19,651,909 | 0.9341 | 0.0974 | 0.1797 | +0.0823 | 1.84 | 0.9753 | 0.9480 |
| MS-TCN | m5c_C912ls | 58,458,757 | 0.9546 | 0.0960 | 0.1756 | +0.0796 | 1.83 | 0.9744 | 0.9486 |
| MS-TCN | m5c_C1680ls | 198,003,973 | 1.0072 | 0.1041 | 0.1770 | +0.0729 | 1.70 | 0.9715 | 0.9477 |

D models: NLL on the C-sample games they trained on vs the C-sample games they did NOT train on (in-distribution):

| model | seen games | NLL seen | unseen games | NLL unseen | NLL held-out fold-0 |
|---|---|---|---|---|---|
