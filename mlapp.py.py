from fastapi import FastAPI
import uvicorn
import pickle

app = FastAPI(debug=True)


@app.get('/')
def home():
    return {'text': 'Insurance pricing prediction solution'}

@app.get('/predict')
def predict(age: int, sex: int, bmi: float, children: int, smoker: int, region: int):
    # Load the model
    model = pickle.load(open('regmodel.pkl', 'rb'))

    # Make predictions using the loaded model
    makeprediction = model.predict([[age, sex, bmi, children, smoker, region]])
    output = round(makeprediction[0], 2)

    return {"The insurance cost is": output}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
