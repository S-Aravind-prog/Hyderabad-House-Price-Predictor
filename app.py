from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

pipe = pickle.load(open('pipe.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    location = request.form['location']
    PropertyType = request.form['PropertyType']
    building_status = request.form['building_status']
    BHK = int(request.form['BHK'])
    #BHK = int(request.form['BHK'])
    area_insqft = float(request.form['area_insqft'])
    
    AreaPerRoom = area_insqft / BHK

    test_input = pd.DataFrame(
        [[ location,
        PropertyType,
        building_status,
        BHK,
        area_insqft,
        AreaPerRoom]],

        columns=  ['location',
        'PropertyType',
        'building_status',
        'BHK',
        'area_insqft',
        'AreaPerRoom'            ]
    ) 

    prediction = pipe.predict(test_input)

    price = prediction[0]

    lower_price = price * 0.90
    upper_price = price * 1.10

    if price >= 100:
        display_price = f"₹{price/100:.2f} Crore"

        lower_price = f"₹{lower_price/100:.2f} Crore"

        upper_price = f"₹{upper_price/100:.2f} Crore"
    else:
        display_price = f"₹{price:.2f} Lakhs"
        lower_price = f"₹{lower_price:.2f} Lakhs"
        upper_price = f"₹{upper_price:.2f} Lakhs"

    if price < 50:
        category = "🏡 Budget House"
    elif price < 150:
        category = "🏠 Mid Range House"
    else:
        category = "🏰 Luxury House"


    return render_template(
        'index.html',

        prediction=display_price,
        lower_price=lower_price,
        upper_price=upper_price,
        category=category,
        location=location,
        propertytype=PropertyType,
        status=building_status,
        bhk=BHK,
        area=area_insqft,
    )

if __name__ == "__main__":
    app.run(debug=True)
