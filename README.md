# pinns-in-mlx

This repo is an exercise in implementing Physics-Informed Neural Networks in MLX. It is still quite work-in-progress. with currently only the 1D Burgers equation implemented. Still to do:

*Verify the implementation (solution, error plots, optimizer)
*Include parameter domains (useful for UQ)
*Compare different optimizers. Currently only the SOAP ("ShampoO with Adam in the Preconditioner's eigenbasis") optimizer is used.

The code can utilized by running ```example_burgers_1d.py```. This produces the following plots:

![loss](https://github.com/user-attachments/assets/006cc5fd-df04-4462-8159-264c281fc5c8)
![prediction](https://github.com/user-attachments/assets/e1248b8b-384f-4c4e-a812-871d6a8348c0)
