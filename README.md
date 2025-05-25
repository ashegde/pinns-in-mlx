# pinns-in-mlx

This repo is an exercise in implementing Physics-Informed Neural Networks in MLX. It is still quite work-in-progress. with currently only the 1D Burgers equation implemented. Still to do:

*Verify the implementation (solution, error plots, optimizer)
*Include parameter domains (useful for UQ)
*Compare different optimizers. Currently only the SOAP ("ShampoO with Adam in the Preconditioner's eigenbasis") optimizer is used.

The code can utilized by running ```example_burgers_1d.py```. This produces the following plots:

![loss](https://github.com/user-attachments/assets/84f7bedd-1045-48bb-b5d4-b71f872d5c70)
![prediction](https://github.com/user-attachments/assets/f4eaa802-c74d-4db8-af50-6082b84ee58a)
