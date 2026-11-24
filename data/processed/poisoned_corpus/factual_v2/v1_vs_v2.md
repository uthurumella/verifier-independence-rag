# factual_v1 vs factual_v2

Code: `ed03d7d Add relevant_llm_full_v1 data card` + `factual_substitution_v2.py` (md5 0227ba028c649f365c4605ce5f0a46ec)
Raw: dev.jsonl md5 1db1a753bd00086e94e3312a6512bec6. v1 rerun reproduces the released factual_v1 byte for byte.

| Check (per question) | v1 | v2 |
|---|---|---|
| questions | 1099 | 1092 |
| pipeline: false value present | 1009 | 999 |
| pipeline: true value present | 35 | 1 |
| own rule: true value present | 40 | 0 |
| impossible date added | 47 | 0 |
| comma lost | 123 | 2 |
| year after 2026 | 9 | 0 |
| false value already in source | 20 | 0 |
| range end changed | 46 | 0 |
| decade broken | 6 | 0 |
| year before 1500 scaled | 10 | 0 |

Questions in both: 1021; same false value: 924; identical passage text: 793.
Only in v1 (dropped by v2): 78 — reasons: {'answer_number_is_part_of_a_decimal': 29, 'number_is_part_of_a_range': 47, 'number_absent_from_gold_passage': 2}
Only in v2 (new): 71
v2 rejected at the check: {'number_is_part_of_a_range': 47, 'answer_number_is_part_of_a_decimal': 29}
v2 passages: 3276 = 1092 questions x 3; passage ids unique: True
v2 split: dev 896 / test 196 (v1: dev 904 / test 195)
v2 types: {'bridge': 1054, 'comparison': 38}
v2 false value redrawn (differs from v1's draw): 167
v2 day-of-month answers: 198; residual: questions with a non-gold passage stating the true number: 377

## Examples where v2 differs from v1

- `5ac240ef55429951e9e684ef` answer '2006': v1 '2006'->'1991' | v2 '2006'->'1991'
- `5a790f5e55429970f5fffe40` answer 'November 15, 1903': v1 '15,'->'30' | v2 '15'->'23'
- `5ac3d68f554299076e296ca6` answer 'August 1973': v1 '1973'->'1968' | v2 '1973'->'1988'
- `5a77ba6c55429967ab10524c` answer 'June 11, 1986': v1 '11,'->'6' | v2 '11'->'6'
- `5abb2f2f554299232ef4a3d7` answer 'October 5, 1930': v1 '5,'->'8' | v2 '5'->'8'
- `5a8a57fe55429930ff3c0da0` answer 'January 24, 2012': v1 '24,'->'17' | v2 '24'->'17'
- `5ac55e835542993e66e82375` answer 'October 25, 1881': v1 '25,'->'40' | v2 '25'->'7'
- `5abe79c755429976d4830b2a` answer '1449': v1 '1449'->'2318' | v2 '1449'->'1464'
