### Wjat is this project about?
Kaggle competitions are essentially about performing Exploretory Data Analysis (EDA) and then training a model for best performance on an appropriate metric. In practice, however, Jupyter notebooks are not enough. 

To be able to experiment, iterate, debug and deploy quickly and easily when working in a team, it is necessary to create a dedicated environment. It is especially important to be able to reproduce the state of a project (code, data, model).
This is the core function of the so-called machine learning devops (short MLOps).
This project can serve as a kind of template for those interested in setting up such an environment and is intended to show the main advantages.

### Who is this content for?
- `Software engineers` looking compare traditional DevOps and MLOps and become even better software engineers.
- `Data scientists` who know that creating jupyter notebooks is not enough in a serious project.
- `College graduates` looking to learn the practical skills they'll need for the industry.

## Directory structure
```bash
app/
├── api.py                    - FastAPI app
└── cli.py                    - CLI app
├── config.py                 - configuration setup

titanic_classification/
├── data.py                   - data processing components
├── eval.py                   - evaluation components
├── main.py                   - training/optimization pipelines
├── models.py                 - model architectures
├── predict.py                - inference components
├── train.py                  - training components
├── feature_engineering.py    - FEng components 
└── utils.py                  - supplementary utilities
```

## Workflows

#### 1. Clone project.
```bash
git clone https://github.com/AnelMusic/mlops_on_titanic_dataset
cd mlops_on_titanic_dataset
```
### 2. Set up environment. (Routine defined in Makefile)
> This will automatically install all dependencies defined in requirements.txt
```bash
make venv
source venv/bin/activate
```
### 3. Download to data (Routine defined in CLI (see app/cli.py))
```bash
dvc pull
```
