Chicken Disease Classification (Coccidiosis Detection)

An end-to-end deep learning project that classifies chicken fecal images as Healthy or Coccidiosis (a common and costly poultry disease), helping farmers catch outbreaks early and reduce flock mortality.

The project uses transfer learning with VGG16, a modular training pipeline orchestrated with DVC, and is served through a Flask web app that can be containerized with Docker.

Overview

Coccidiosis is one of the most economically damaging diseases in poultry farming. Early detection from a simple photo of chicken droppings can help farmers isolate affected birds and start treatment before the disease spreads. This project builds a CNN-based image classifier (VGG16 backbone) that takes a fecal sample image and predicts whether the sample is Healthy or shows signs of Coccidiosis.

The codebase follows a production-style, modular ML project structure — config-driven, pipeline-based, and deployable — rather than a single notebook script.

How It Works

The model uses transfer learning: a VGG16 network pre-trained on ImageNet is used as a frozen feature extractor, with a custom classification head trained on the chicken fecal image dataset.

Parameter	Value
Base model	VGG16 (ImageNet weights)
Image size	224 x 224 x 3
Classes	2 (Healthy, Coccidiosis)
Batch size	16
Data augmentation	Enabled
Learning rate	0.01

These are all configurable in params.yaml without touching the code.

Project Architecture / Pipeline

The training workflow is broken into independent, reproducible stages managed by DVC (dvc.yaml):

1. Data Ingestion        → Downloads & extracts the chicken fecal image dataset
2. Prepare Base Model    → Loads VGG16, removes the top layer, attaches a custom classifier head
3. Training               → Trains the model with augmentation, saves model.h5
4. Evaluation             → Evaluates the trained model and logs metrics to scores.json

Each stage is a standalone pipeline script under src/cnnClassifier/pipeline/, so any stage can be re-run in isolation, and DVC will only re-run stages whose dependencies changed.

Project Structure
Chicken-Disease-Classification/
├── config/
│   └── config.yaml            # Paths & artifact locations for each pipeline stage
├── src/cnnClassifier/
│   ├── components/            # Core logic (data ingestion, model prep, training, callbacks)
│   ├── config/                # Configuration manager
│   ├── constants/              # Global constants (paths to config/params files)
│   ├── entity/                 # Config data classes
│   ├── pipeline/               # stage_01 → stage_04 pipeline scripts
│   └── utils/                  # Common helper functions
├── templates/                  # HTML templates for the Flask web app
├── .github/workflows/          # CI/CD workflow
├── app.py                      # Flask application (prediction API + UI)
├── main.py                     # Runs the full training pipeline end-to-end
├── dvc.yaml                    # DVC pipeline stage definitions
├── params.yaml                 # Model/training hyperparameters
├── scores.json                 # Evaluation metrics (loss/accuracy)
├── setup.py                    # Packages the project as an installable module
├── template.py                 # Script to scaffold the project structure
└── requirements.txt            # Python dependencies
Tech Stack
Python, TensorFlow / Keras — model building & training
VGG16 — pretrained CNN backbone (transfer learning)
DVC — pipeline versioning & reproducibility
Flask, Flask-Cors — web app / inference API
PyYAML, python-box, ensure — config management
GitHub Actions — CI/CD workflow
Getting Started
1. Clone the repository
bash
git clone https://github.com/ChaitaliJaibhaye/Chicken-Disease-Classification.git
cd Chicken-Disease-Classification
2. Create a virtual environment & install dependencies
bash
conda create -n chicken python=3.8 -y
conda activate chicken

pip install -r requirements.txt
3. Run the training pipeline

This runs all four DVC stages (data ingestion → base model prep → training → evaluation):

bash
python main.py

Or reproduce only the stages whose inputs changed, using DVC:

bash
dvc repro
4. Launch the web app
bash
python app.py

Then open http://localhost:8080 (or the port shown in the console) in your browser, upload a chicken fecal image, and get a prediction.

Evaluation

After training, model performance (loss/accuracy on the validation set) is written to scores.json by the evaluation stage, so results are versioned alongside the code and can be tracked with dvc metrics show.

Configuration

All tunable settings live in two files, so you can experiment without editing source code:

config/config.yaml — file paths, artifact directories, dataset source URLs
params.yaml — model hyperparameters (image size, batch size, epochs, learning rate, augmentation, etc.)
Contributing

Contributions, issues, and feature requests are welcome!

Fork the project
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
License

This project is licensed under the MIT License. See the LICENSE file for details.

cknowledgements
VGG16 architecture — Simonyan & Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition
Chicken fecal image dataset used for Coccidiosis vs. Healthy classification
