# pinns-in-mlx

This repo is an exercise in implementing Physics-Informed Neural Networks in MLX. It is still quite work-in-progress. with currently only the 1D Burgers equation implemented. Still to do:

* Verify the implementation (solution, error plots, optimizer)
* Include parameter domains (useful for UQ)

The code can utilized by running ```example_burgers_1d.py```. This produces the following plots:

Training losses for different methods. Recall, new training data is sampled during each iteration:
![loss_combined](https://github.com/user-attachments/assets/f4abbd21-eead-47e8-b97c-af8012fa5c86)

Field evolution associated with the SOAP optimizer:
![prediction](https://github.com/user-attachments/assets/35e43f2a-6f9c-46da-b9e2-a4c5ba772014)
