# Reference-style full-input 200+200 observation

Status: **complete**.

Reference-style adaptation of MixKV Figure 1(b), not an exact reproduction: Qwen2.5 text and vision-language models, fixed LongBench qasper and OCRBench inputs. Models, datasets, input lengths and semantic roles differ; comparison does not isolate a within-model modality effect or establish merge safety.

All distinct-token-pair cosine over the full native prefill sequence in each input; self-pairs excluded; signed values retained

Equal KV-head and layer weights within each input, then equal weights for the 200 prescribed inputs per model; no token-length weighting

One layer/native-KV-head mean over the 200 inputs; 28*4=112 descriptive means per model; not independent observations

Exploratory 95% percentile source-cluster bootstrap for independently resampled groups; fixed source rows, no causal or head-as-sample inference. CI unavailable if either required group has fewer than two source clusters.

| Quantity | Text model (200 inputs) | Vision-language model (200 inputs) | Vision − text |
|---|---:|---:|---:|
| k_raw | 0.677931 | 0.592037 | -0.085894 [-0.089431, -0.082514] |
| k_rope | 0.498643 | 0.574448 | 0.075805 [0.072580, 0.078912] |
| v | 0.096660 | 0.130154 | 0.033494 [0.030272, 0.036592] |

The main figure uses post-RoPE K and V; raw K is provided separately. Self-pairs are excluded; negative values and opposite-direction results are retained.

| Group | Model | Revision | Valid document IDs | Source clusters | Input token min / max / mean |
|---|---|---|---|---|---|
| text | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199] | 148 | 2094 / 22064 / 5211.4 |
| vision | Qwen/Qwen2.5-VL-7B-Instruct | cc594898137f460bfe9f0759e9844b3ce807cfb5 | [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695, 700, 705, 710, 715, 720, 725, 730, 735, 740, 745, 750, 755, 760, 765, 770, 775, 780, 785, 790, 795, 800, 805, 810, 815, 820, 825, 830, 835, 840, 845, 850, 855, 860, 865, 870, 875, 880, 885, 890, 895, 900, 905, 910, 915, 920, 925, 930, 935, 940, 945, 950, 955, 960, 965, 970, 975, 980, 985, 990, 995] | 199 | 61 / 16308 / 1076.3 |
