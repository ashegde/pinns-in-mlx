# pinns-in-mlx

This repo is an exercise in implementing Physics-Informed Neural Networks in MLX. It is still quite work-in-progress. with currently only the 1D Burgers equation implemented. Still to do:

*Verify the implementation (solution, error plots, optimizer)
*Include parameter domains (useful for UQ)
*Compare different optimizers. Currently only the SOAP ("ShampoO with Adam in the Preconditioner's eigenbasis") optimizer is used.

The code can utilized by running ```example_burgers_1d.py```. This produces the following plots:

![loss](https://github.com/user-attachments/assets/3ef0f314-2ecf-49ba-8b00-4450c9947756)
![prediction](https://github.com/user-attachments/assets/2d5e88ef-ff92-40c5-b59b-09baf89eb019)

