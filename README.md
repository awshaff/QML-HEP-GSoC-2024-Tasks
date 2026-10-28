# QML-HEP GSoC 2024 – Evaluation Tasks

My solutions to the evaluation test of the ML4SCI **Quantum Machine Learning for High Energy Physics (QML-HEP)** project for Google Summer of Code 2024. The test has three parts: a quantum computing warm-up, a classical graph neural network task, and a short open commentary.

> **Status: partially complete.** Task I and Task III are done. Task II is only partly done (one of the two requested architectures, and a known evaluation issue). Details are in [Status and known issues](#status-and-known-issues).

## Repository layout

| Path | Contents |
|---|---|
| `Task I Quantum Computing Part/` | Quantum circuits in PennyLane |
| `Task II Classical Graph Neural Network (GNN)/` | ParticleNet-Lite for quark/gluon jet classification |
| `QML-HEP-GSoC-2024-Tasks.ipynb` | Combined notebook: Task I, Task III, and the start of Task II (data loading only; the actual Task II work lives in its own folder) |
| `requirements*.txt` | Dependencies (generic, macOS, Ubuntu, Windows 10) |

## Task I – Quantum computing (PennyLane)

1. **Five-qubit circuit.** Hadamard on every qubit, CNOTs on (0,1), (1,2), (2,3), (3,4), SWAP(0,4), and an RX(π/2) rotation, drawn with `qml.draw_mpl`.
2. **Swap test.** Qubits 1–2 hold H and RX(π/3) states, qubits 3–4 hold Hadamard states, and an ancilla runs the swap test via two controlled-SWAPs.

**Result.** With 1024 shots, the ancilla probabilities were about `[0.74, 0.26]` to `[0.75, 0.25]` across runs. This matches theory: the overlap between the two states is 0.5, so P(ancilla = 0) = (1 + 0.5) / 2 = 0.75.

## Task II – Quark/gluon jet classification with a graph network

**Data.** The [ParticleNet quark/gluon jet dataset](https://zenodo.org/records/3164691): 100,000 jets (50k quark, 50k gluon), each a zero-padded point cloud of up to 139 particles with four features (pt, rapidity, azimuthal angle, PDG ID). I split it 70/15/15 into train, validation, and test sets.

**Model.** ParticleNet-Lite from the [ParticleNet reference implementation](https://github.com/hqucms/ParticleNet) (`tf_keras_model.py`), which I used as-is rather than writing the architecture myself. Each jet is treated as a point cloud: the (rapidity, azimuth) pair is passed as the coordinates, all four particle features are passed as node features, and a mask marks real particles versus padding. The neighbourhood graph is built inside the model by its EdgeConv blocks.

**Training.** Categorical cross-entropy, Adam, batch size 1024, 100 epochs, best-validation-accuracy checkpointing. Trained on an Apple M2 (Metal), about 2 minutes per epoch.

| Metric | Value |
|---|---|
| Best validation accuracy | 0.812 (epoch 78) |
| Test accuracy | Re-evaluation pending, see below |

## Task III – Open task

A short written commentary on quantum computing and quantum machine learning, in the combined notebook.

## Status and known issues

I'm documenting these openly rather than hiding them:

- **Only one graph architecture.** The task asks for two architectures, a description of how the point cloud is turned into nodes and edges, and a discussion of their performance. I only implemented ParticleNet-Lite. There is no second architecture and no written comparison.
- **Test evaluation bug.** In the Task II notebook, the test mask is overwritten by the validation mask (`mask_test = mask_val.reshape(...)`). This is why the notebook reports a test accuracy of 0.756 against a validation accuracy of 0.812. That test number is not reliable. The fix is to build the mask from `x_test` and re-run the evaluation; I will update the table above when done.
- **Learning-rate schedule.** The `if epoch > 10 ... elif epoch > 20` ordering means the second decay never triggers, so the learning rate stays at 1e-4 after epoch 10.
- **Final-epoch evaluation.** The notebook evaluates the model after epoch 100, not the best checkpoint (epoch 78).
- **Metrics.** Only accuracy is reported; AUC and ROC curves, the standard metrics for jet tagging, are missing.
- **Combined notebook.** The Task II cells there are leftover exploration (only the first particle of each jet is loaded) and do not contain the solution.

## Running the code

```bash
# pick the requirements file that matches your platform
pip install -r requirements.txt

# Task II needs the dataset (about 100k jets) in ./data/QG_jets.npz
mkdir -p data
wget -P data https://zenodo.org/record/3164691/files/QG_jets.npz
```

Then open the notebooks with Jupyter. The Task II notebook imports `tf_keras_model`, so that file needs to be on the Python path (see Model above for its source).

## Acknowledgements

ParticleNet architecture and dataset: H. Qu and L. Gouskos, *ParticleNet: Jet Tagging via Particle Clouds*, Phys. Rev. D 101, 056019 (2020).
