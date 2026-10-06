# Backpropagation for Feedforward Networks

Seminar project for the PhD course **Artificial Neural Networks (20.IDI23)**, Doctoral Academic Studies in Computer Science, Faculty of Sciences and Mathematics, University of Niš.
Instructor: Prof. Branimir Todorović · Author: Elvir Muslić

The notebook implements the forward pass and the backward recursion of backpropagation by hand, with PyTorch tensors in double precision, trains two small networks with plain gradient descent, and compares the hand-written gradients with automatic differentiation and with central finite differences. It has 11 checks, which stop the run if any checked value deviates; the last one compares every number quoted in the paper and below with the run. All random seeds are fixed.

## Results

- **Worked training step** (one hidden unit, one output unit): the nine hand-computed values of the paper's Table 1 are reproduced to the printed digits; autograd agrees with the hand-written gradients to 1.4e-17 and a central finite difference to 2.9e-11.
- **4-8-8-3 network** (139 parameters): the hand-written recursion matches autograd to a relative difference of 1.9e-16 and central finite differences to 5.4e-10.
- **XOR** (2-4-1 network, gradient descent with step size 1): the loss falls from 0.128 to 4.2e-4 in 2000 steps and decreases at every one of the first 100; all four inputs are classified correctly.
- **Two moons** (200 training and 200 test points, 2-16-1 network, 5000 full-batch steps): training accuracy 1.0, test accuracy 0.995, final loss 0.0010.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace backprop_project.ipynb
```

- Tested with Python 3.14. The reported run took 1.3 s on one CPU thread.
- The run rewrites `figures/` and `results.json`.

## Files

| File | Contents |
|---|---|
| `backprop_project.ipynb` | the notebook, with outputs |
| `results.json` | the key numbers written by the run |
| `figures/` | the two figures: the training loss per step, and the moons with the decision boundary |
| `requirements.txt` | pinned package versions |
