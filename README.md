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

## Basic Workflows

#### 1. Clone project.
```bash
git clone https://github.com/AnelMusic/mlops_on_titanic_dataset
cd mlops_on_titanic_dataset
```
#### 2. Set up environment. (Routine defined in Makefile)
> This will automatically install all dependencies defined in requirements.txt
```bash
make venv
source venv/bin/activate
```
#### 3. Download to data (Routine defined in CLI (see app/cli.py))
```bash
dvc pull
```
#### 4. Evaluate the (currently) best model on a test dataset
> This will automatically load the best model (with respect to accuracy) from all MLflow experint runs
```bash
titanic_classification eval-model
```
The expected output should look as follows:

![eval_screenshot](https://user-images.githubusercontent.com/32487291/133000819-cc1ab06c-5e2c-42b7-bbc6-936331e519c5.png)


#### 4. (Optional) Interact with the model using RESTAPI
> We're using Uvicorn, a fast ASGI server to launch our application. 
> The API will run on localhost port 5000 ---> http://0.0.0.0:5000
> Hint: Use http://0.0.0.0:5000/docs to access the interactive documentation
```bash
make api
```
You can ask for the model parameters using the /model_params endpoint. Try it out directly using the interactive documentation:

![api_screenshot](https://user-images.githubusercontent.com/32487291/133000727-825ea695-2eb3-4ea9-a7a3-d21dad44b4d1.png)

 

## Basic Workflows



