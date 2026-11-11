| family | model | parameters | training time (pff0) | PFF f0 fine3 (12) | diff vs shipped | SKC fine3 (20) | diff vs shipped | Metrica dev | diff vs shipped (2 games) | loss logged | scoring path / check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pointer | r5_ptr_ballc | 796,144 | 11 min | 0.8083 | -0.0134 ± 0.0041 (2/12) | 0.5879 | -0.0023 ± 0.0032 (8/20) | 0.7650 | +0.0034 (1/2) | train team/ptr 0.2461/0.1889 (it 6000) | earlier screens |
| pointer | r5_ptr_ballc_p20M (ref) | 19,258,736 | 26 min | 0.8217 |  | 0.5902 |  | 0.7616 |  | train team/ptr 0.2091/0.1072 | earlier screens |
| pointer | r5_ptr_ballc_p180M | 178,146,856 | 79 min | 0.8213 | -0.0004 ± 0.0032 (5/12) | 0.5823 | -0.0080 ± 0.0031 (7/20) | 0.7604 | -0.0012 (1/2) | train team/ptr 0.2080/0.1034 | CUDA (pod); numpy check RUNNING (Mac, ~3 h/game) |
| pointer | r5_ptr_ballc_p500M | 497,550,304 | 64 min | 0.8157 | -0.0060 ± 0.0038 (5/12) | 0.5820 | -0.0082 ± 0.0037 (7/20) | 0.7599 | -0.0017 (1/2) | train team/ptr 0.2174/0.1225 | CUDA (pod); numpy check RUNNING (pod CPU + Mac, ~8 h/game); 3 SKC games rerun after GPU OOM |
| tcn | o_ls05 (ref) | 215,812 | 1 min (MPS) | 0.7903 |  | 0.5850 |  | 0.7316 |  | train 0.2720 (it 4000) | earlier screens |
| tcn | o_ls05_C208 | 2,139,076 | 3 min | 0.7913 | +0.0010 ± 0.0052 (8/12) | 0.5859 | +0.0009 ± 0.0030 (8/20) | 0.7308 | -0.0009 (0/2) | train 0.2497 | earlier screens |
| tcn | o_ls05_C640 | 19,852,804 | 5 min | 0.7951 | +0.0048 ± 0.0045 (9/12) | 0.5779 | -0.0071 ± 0.0028 (7/20) | 0.7416 | +0.0100 (2/2) | train 0.2324 | earlier screens |
| tcn | o_ls05_C2032 | 198,802,756 | 22 min | 0.7989 | +0.0086 ± 0.0039 (9/12) | 0.5837 | -0.0013 ± 0.0036 (8/20) | 0.7321 | +0.0004 (1/2) | train 0.2283 | numpy CPU (exact) |
| tcn | o_ls05_C3232 | 502,369,156 | 14 min | 0.7933 | +0.0030 ± 0.0039 (9/12) | 0.5738 | -0.0113 ± 0.0044 (6/20) | 0.7366 | +0.0050 (2/2) | train 0.2305 | CUDA, check SAME (3 SKC games on the numpy CPU path after a GPU OOM) |
| mstcn | m5c_C192ls (ref) | 2,630,317 | 5 min | 0.7944 |  | 0.5558 |  | 0.6947 |  | train 1.0138 (ep 39) | earlier screens |
| mstcn | m5c_C528ls | 19,651,909 | 9 min | 0.8021 | +0.0077 ± 0.0035 (9/12) | 0.5556 | -0.0001 ± 0.0032 (9/20) | 0.6862 | -0.0085 (0/2) | train 0.9341 | earlier screens |
| mstcn | m5c_C912ls | 58,458,757 | 17 min | 0.7993 | +0.0048 ± 0.0031 (8/12) | 0.5542 | -0.0016 ± 0.0030 (9/20) | 0.6906 | -0.0040 (1/2) | train 0.9546 | earlier screens |
| mstcn | m5c_C1680ls | 198,003,973 | 46 min | 0.7988 | +0.0043 ± 0.0041 (7/12) | 0.5560 | +0.0002 ± 0.0039 (11/20) | 0.6944 | -0.0003 (1/2) | train 1.0072 | CUDA, check SAME |
| mstcn | m5c_C2672ls | 500,464,277 | 22 min | 0.0042 | -0.7903 ± 0.0077 (0/12) | - | - | 0.0049 | -0.6898 (0/2) | DIVERGED: train 9.0e6 at ep 1, 2534 at ep 39 | CUDA, check SAME (numpy also 0.00) |
| asf | b2_asformer_d128L9 | 3,627,474 | 33 min | 0.7840 | +0.0132 ± 0.0049 (8/12) | 0.5459 | +0.0111 ± 0.0030 (16/20) | 0.6988 | +0.0063 (2/2) | val_ce 0.1462 (best ep 26) | earlier screens |
| asf | b5_asformer_d256L20 (ref) | 31,817,547 | 70 min | 0.7709 |  | 0.5348 |  | 0.6926 |  | val_ce 0.1595 (ep 26) | earlier screens |
| asf | b6_asformer_d640L20 | 197,508,555 | 52 min | 0.7909 | +0.0200 ± 0.0039 (10/12) | 0.5419 | +0.0071 ± 0.0023 (18/20) | 0.6923 | -0.0003 (1/2) | val_ce 0.1415 (ep 21) | PFF f0 numpy; SKC CUDA SAME; Met MPS SAME; folds 1-3 MPS SAME |
| asf | b7_asformer_d1024L20 | 504,757,323 | 87 min | 0.7975 | +0.0266 ± 0.0047 (12/12) | 0.5403 | +0.0054 ± 0.0030 (13/20) | 0.6888 | -0.0037 (0/2) | val_ce 0.1368 (ep 22) | CUDA (pod); numpy check RUNNING (Mac, ~2 h) |
| radar | radar_pff (ref) | 111,058 | 446 s net (MPS) | 0.7175 |  | 0.5848 |  | 0.7343 |  | LGB pff0 d/ha/c 0.1153/0.1725/0.0888 | earlier screens |
| radar | radar_pff_w10 | 1,139,650 | 365 s net | 0.7364 | +0.0189 ± 0.0043 (12/12) | 0.5933 | +0.0085 ± 0.0024 (14/20) | 0.7340 | -0.0002 (1/2) | LGB pff0 d/ha/c 0.1067/0.1616/0.0854 | numpy CPU (exact) |
| radar | radar_pff_w100 | 11,235,330 | 780 s net | 0.7408 | +0.0233 ± 0.0053 (11/12) | 0.5959 | +0.0111 ± 0.0022 (18/20) | 0.7329 | -0.0014 (1/2) | - | coordinator |

- r5_ptr_ballc PFF all 48 (OOF, fold models): 0.8137  vs r5_ptr_ballc_p20M 0.8270: -0.0133 ± 0.0018 (9/48)
- b2_asformer_d128L9 PFF all 48 (OOF, fold models): 0.7940  vs b5_asformer_d256L20 0.7762: +0.0178 ± 0.0022 (40/48)
- b6_asformer_d640L20 PFF all 48 (OOF, fold models): 0.8004  vs b5_asformer_d256L20 0.7762: +0.0242 ± 0.0019 (45/48)
