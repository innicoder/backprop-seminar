# Backpropagation and Recurrent Networks: Implementation and Verification

Seminar project for the PhD course **Artificial Neural Networks (20.IDI23)**, Doctoral Academic Studies in Computer Science, Faculty of Sciences and Mathematics, University of Niš.
Instructor: Prof. Branimir Todorović · Author: Elvir Muslić

The notebook implements the backward recursion of backpropagation and backpropagation through time (BPTT) by hand, and checks each against PyTorch's automatic differentiation. All random seeds are fixed.

## Results

- **Backpropagation:**
  - On a one-hidden-unit network, the hand-written gradients match autograd to 2.8e-17 and a central finite difference to 2.9e-11.
  - On a 4-8-8-3 network, the largest relative difference to autograd is 4.1e-16.
- **BPTT:** the hand-written gradient of a recurrent network matches autograd exactly (relative difference 0).
- **Vanishing and exploding gradients:**
  - Jacobian products shrink or grow like ρ^k with the distance k in time.
  - With saturated units, even ρ = 1.1 decays by a factor of 0.475 per step.
- **LSTM:** with a forget-gate bias of 8, the cell path keeps 0.98 of the gradient over 60 steps.
- **Adding problem** (T = 50, 3000 Adam steps):
  - A plain recurrent network stays at the baseline: test MSE 0.1753, against a baseline of 0.1754.
  - The LSTM reaches 1.3e-4.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace backprop_rnn_project.ipynb
```

- Tested with Python 3.14. The reported run took about 54 s on one CPU thread.
- The run rewrites `figures/` and `results.json`.

## Files

| File | Contents |
|---|---|
| `backprop_rnn_project.ipynb` | the notebook, with outputs |
| `results.json` | the key numbers written by the run |
| `figures/` | the four figures |
| `requirements.txt` | pinned package versions |
