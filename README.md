# Face regonition with occlusion

## Branch Naming Convention

In our project, we follow a clear branch naming strategy to maintain consistency and readability across all sprint and development activities.
Below is a list of branch types and their purposes.

|Branch Type | Purpose	| Example |
-------------|----------| ------- |
|feature/ | Used for developing new features or functionalities. | feature/sprint3-user-auth |
|bugfix/  | Used for fixing bugs discovered during development or testing. | bugfix/sprint3-login-error |
|test/	 | Used for experiments, prototypes, or proof-of-concept (POC) work. | test/sprint3-model-eval |
|merge/	 | Used for combining several experimental or feature branches.	| merge/sprint3-merge-experiments |
|release/ | Used for preparing the final version of a sprint before deployment. | release/sprint3 |
|hotfix/  | Used for urgent fixes applied after a release. | hotfix/fix-typo-readme |

project/
│
├── train_model.py
├── evaluate_model.py   ← SKRYPT DO EWALUACJI
├── models/
│   └── face_model.pt
├── data/
│   ├── test_no_occlusion/
│   └── test_occlusion/
└── results/
    ├── metrics.csv
    └── tsne_plot.png
