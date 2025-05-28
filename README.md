# pinns-in-mlx

This repo is an exercise in implementing Physics-Informed Neural Networks in MLX. It is still quite work-in-progress. with currently only the 1D Burgers equation implemented. Still to do:

* Verify the implementation (solution, error plots, optimizer)
* Include parameter domains (useful for UQ)
* Compare different optimizers. Currently only the SOAP ("ShampoO with Adam in the Preconditioner's eigenbasis") optimizer is used.

The code can utilized by running ```example_burgers_1d.py```. This produces the following plots:

Training losses for different methods. Recall, new training data is sampled during each iteration:
![loss_combined](https://github.com/user-attachments/assets/5f9f2943-dd18-49a7-9aab-115b23a52c6b)

Field evolution associated with the SOAP optimizer:
![prediction](https://github.com/user-attachments/assets/5552f9c6-b028-482c-b8b9-0beece9f43f4)
