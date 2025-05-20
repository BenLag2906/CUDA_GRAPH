Recently, with the advent of deepseek, time of training and
inference, and generally time processing have been considered as a
major constraint. For example, to limit exchange between CPU
memory and GPU memory, some architecture could be developed.
However, many optimizations could be developed in CUDA
manipulations. In the following description, we focus on
optimisation to limit communication between CPU and GPU.

![graph with kernel](https://github.com/user-attachments/assets/f46e5f3d-eeba-494e-bf91-83b1660e4c81)
