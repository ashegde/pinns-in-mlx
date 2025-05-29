# pinns-in-mlx

This repo is an exercise in implementing Physics-Informed Neural Networks in MLX. It is still quite work-in-progress. with currently only the 1D Burgers equation implemented. Still to do:

* Verify the implementation (solution, error plots, optimizer)
* Hyperparameter optimization for each experiment (independently)
* Include parameter domain in training, which is useful for tasks such as UQ.

The code can utilized by running ```example_burgers_1d.py```. This produces the following plots:

Training losses for different methods. Recall, new training data is sampled during each iteration. In these experiments, weight decay is negligible, hence Adam and AdamW have similar results.
![loss_combined](https://github.com/user-attachments/assets/e4b501ac-bdc7-419a-bc7b-17b4e5033fff)

Field evolution associated with the SOAP optimizer:
![prediction](https://github.com/user-attachments/assets/d2a73356-5894-44f9-b8f1-aad396c4c219)

