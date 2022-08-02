from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()
pickle_in = open("dec_tree.pkl","rb")
dec_tree = pickle.load(pickle_in)

class Titanic(BaseModel):
    sex: float 
    p_class: float 
    age: float 
    sibsp: float
    parch: float
    embarked: float

@app.get("/")
def root():
    return {"Message":"Hello World"}

@app.post('/predict')
def predict(data: Titanic):
    data = data.dict()
    sex = data['sex']
    p_class = data['p_class']
    age = data['age']
    sibsp = data['sibsp']
    parch = data['parch']
    embarked = data['embarked']
    pred = dec_tree.predict([[sex, p_class, age, sibsp, parch, embarked]])
    print(pred[0])
    return {"Survive ? "+ str(pred[0])}