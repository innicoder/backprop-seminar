# Backpropagation for Feedforward Networks

A seminar (study) project for the PhD course **Artificial Neural Networks (20.IDI23)**, Doctoral Academic Studies in Computer Science, Faculty of Sciences and Mathematics, University of Niš. It is course work, written to learn the method, not research.
Instructor: Prof. Branimir Todorović · Author: Elvir Muslić

The notebook accompanies the seminar paper. It computes one training step of the smallest network by hand and compares the gradient with PyTorch's automatic differentiation and with central finite differences. It then trains two small networks built from `torch.nn` layers: `loss.backward()` computes the gradient by automatic differentiation, which carries out the backward recursion, and `torch.optim.SGD` applies plain gradient descent. The two-moons data come from scikit-learn's `make_moons`. All computations are in double precision, and the random seeds are fixed.

## Results

- **Worked training step** (one hidden unit, one output unit): the nine values of the paper's Table 1, printed to nine decimals; autograd agrees with the hand-computed gradients to 1.4e-17 and a central finite difference to 2.9e-11.
- **XOR** (2-4-1 network, step size 1, 2000 steps): the loss falls from 0.126 to 4.6e-4; the outputs are 0.03, 0.98, 0.97 and 0.04, so all four inputs are classified correctly.
- **Two moons** (200 training and 200 test points from `make_moons` with seeds 0 and 1, i.e. the same two curves with different noise; 2-16-1 network, step size 1, 5000 full-batch steps): the loss falls from 0.1241 to 0.0017; training accuracy 1.0, test accuracy 1.0.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace backprop_project.ipynb
```

Or open `backprop_project.ipynb` in VS Code, select the `.venv` kernel and choose Run All.

- Tested with Python 3.14.7. The run takes a few seconds.
- The run rewrites `figures/`.

## Files

| File | Contents |
|---|---|
| `backprop_project.ipynb` | the notebook, with outputs |
| `figures/` | the two figures: the training loss per step, and the moons with the decision boundary |
| `requirements.txt` | pinned package versions |
