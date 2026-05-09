from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

app = Flask(__name__)

def get_db_connection():
    """
    Creates a PostgreSQL database connection using DATABASE_URL.
    """
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """
    Creates the predictions table if it does not already exist.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_predictions (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                age NUMERIC,
                gender_encoded INTEGER,
                self_employed INTEGER,
                family_history INTEGER,
                work_interfere INTEGER,
                no_employees INTEGER,
                tech_company INTEGER,
                benefits INTEGER,
                care_options INTEGER,
                wellness_program INTEGER,
                seek_help INTEGER,
                anonymity INTEGER,
                leave_value INTEGER,
                mental_health_consequence INTEGER,
                physical_health_consequence INTEGER,
                coworkers INTEGER,
                supervisor INTEGER,
                mental_vs_physical INTEGER,
                sentiment_encoded INTEGER,

                prediction_result TEXT
            );
        """)

        conn.commit()
        cur.close()
        conn.close()

        print("PostgreSQL table is ready.")

    except Exception as e:
        print(f"Error initializing PostgreSQL database: {e}")
        
init_db()        
# Load your pre-trained model
model = joblib.load('rfmodel.pkl')

# Define encoding strategies for categorical variables
encoding_strategies = {
    'Gender_encoded': {'Male': 0, 'Female': 1, 'Others': 2},
    'self_employed': {'Yes': 1, 'No': 0},
    'family_history': {'Yes': 1, 'No': 0},
    'work_interfere': {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3},
    'no_employees': {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5},
    'tech_company': {'Yes': 1, 'No': 0},
    'benefits': {'Yes': 1, 'No': 0, "Don't know": 2},
    'care_options': {'Yes': 1, 'No': 0, 'Not sure': 2},
    'wellness_program': {'Yes': 1, 'No': 0, "Don't know": 2},
    'seek_help': {'Yes': 1, 'No': 0, "Don't know": 2},
    'anonymity': {'Yes': 1, 'No': 0, "Don't know": 2},
    'leave': {'Very easy': 0, 'Somewhat easy': 1, 'Somewhat difficult': 3, "Don't know": 2},
    'mental_health_consequence': {'Yes': 1, 'No': 0, 'Maybe': 2},
    'physical_health_consequence': {'Yes': 1, 'No': 0, 'Maybe': 2},
    'coworkers': {'Yes': 1, 'No': 0, 'Some of them': 2},
    'supervisor': {'Yes': 1, 'No': 0, 'Some of them': 2},
    'mental_vs_physical': {'Yes': 1, 'No': 0, "Don't know": 2},
    'sentiment_encoded': {'Positive': 1, 'Negative': -1, 'Neutral': 0}
}



# Function to save user data into postgresql
def save_to_postgresql(data, prediction_result):
    """
    Save user form data and prediction result into PostgreSQL.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        user_id = str(datetime.now().timestamp())

        cur.execute("""
            INSERT INTO user_predictions (
                user_id,
                age,
                gender_encoded,
                self_employed,
                family_history,
                work_interfere,
                no_employees,
                tech_company,
                benefits,
                care_options,
                wellness_program,
                seek_help,
                anonymity,
                leave_value,
                mental_health_consequence,
                physical_health_consequence,
                coworkers,
                supervisor,
                mental_vs_physical,
                sentiment_encoded,
                prediction_result
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            );
        """, (
            user_id,
            float(data.get('Age', 0)),

            encoding_strategies['Gender_encoded'].get(data.get('Gender_encoded')),
            encoding_strategies['self_employed'].get(data.get('self_employed')),
            encoding_strategies['family_history'].get(data.get('family_history')),
            encoding_strategies['work_interfere'].get(data.get('work_interfere')),
            encoding_strategies['no_employees'].get(data.get('no_employees')),
            encoding_strategies['tech_company'].get(data.get('tech_company')),
            encoding_strategies['benefits'].get(data.get('benefits')),
            encoding_strategies['care_options'].get(data.get('care_options')),
            encoding_strategies['wellness_program'].get(data.get('wellness_program')),
            encoding_strategies['seek_help'].get(data.get('seek_help')),
            encoding_strategies['anonymity'].get(data.get('anonymity')),
            encoding_strategies['leave'].get(data.get('leave')),
            encoding_strategies['mental_health_consequence'].get(data.get('mental_health_consequence')),

            # fallback added in case your HTML field has older name
            encoding_strategies['physical_health_consequence'].get(
                data.get('physical_health_consequence') or data.get('phys_health_consequence')
            ),

            encoding_strategies['coworkers'].get(data.get('coworkers')),
            encoding_strategies['supervisor'].get(data.get('supervisor')),
            encoding_strategies['mental_vs_physical'].get(data.get('mental_vs_physical')),
            encoding_strategies['sentiment_encoded'].get(data.get('sentiment_encoded')),

            prediction_result
        ))

        conn.commit()
        cur.close()
        conn.close()

        print("Data saved to PostgreSQL successfully.")

    except Exception as e:
        print(f"Error saving data to PostgreSQL: {e}")
        
@app.route('/index')
def index():
    # Define dynamic fields and their options
    dynamic_fields = {
        'benefits': ['Yes', 'No', "Don't know"],
        'care_options': ['Yes', 'No', 'Not sure'],
        'wellness_program': ['Yes', 'No', "Don't know"],
        'seek_help': ['Yes', 'No', "Don't know"],
        'anonymity': ['Yes', 'No', "Don't know"],
        'leave': ['Very easy', 'Somewhat easy', 'Somewhat difficult', "Don't know"],
        'mental_health_consequence': ['Yes', 'No', 'Maybe'],
        'physical_health_consequence': ['Yes', 'No', 'Maybe'],
        'coworkers': ['Yes', 'No', 'Some of them'],
        'supervisor': ['Yes', 'No', 'Some of them'],
        'mental_vs_physical': ['Yes', 'No', "Don't know"]
    }
    return render_template('index.html', dynamic_fields=dynamic_fields, prediction=None)

@app.route('/')
def home():
    return render_template('home.html')
    
def fetch_data_from_postgresql():
    """
    Fetch all prediction records from PostgreSQL.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT
                user_id,
                created_at AS timestamp,
                age AS "Age",
                gender_encoded AS "Gender_encoded",
                self_employed,
                family_history,
                work_interfere,
                no_employees,
                tech_company,
                benefits,
                care_options,
                wellness_program,
                seek_help,
                anonymity,
                leave_value AS leave,
                mental_health_consequence,
                physical_health_consequence,
                coworkers,
                supervisor,
                mental_vs_physical,
                sentiment_encoded,
                prediction_result
            FROM user_predictions
            ORDER BY created_at DESC;
        """)

        items = cur.fetchall()

        cur.close()
        conn.close()

        return [dict(item) for item in items]

    except Exception as e:
        print(f"Error fetching data from PostgreSQL: {e}")
        return []


@app.route('/view_data')
def view_data():
    # Fetch raw data from postgresql
    dynamo_data = fetch_data_from_postgresql()

    # Analyze data for charts
    prediction_counts = {'Needs Treatment': 0, 'No Treatment Needed': 0}
    age_distribution = []
    gender_distribution = {'Male': 0, 'Female': 0, 'Others': 0}
    work_interfere_levels = {'Never': 0, 'Rarely': 0, 'Sometimes': 0, 'Often': 0}
    benefits_vs_prediction = {'Yes': {'Needs Treatment': 0, 'No Treatment Needed': 0},
                              'No': {'Needs Treatment': 0, 'No Treatment Needed': 0},
                              "Don't know": {'Needs Treatment': 0, 'No Treatment Needed': 0}}

    # Decoding and analysis
    for item in dynamo_data:
        # Decode Gender
        gender_code = item.get('Gender_encoded', None)

        gender = next(
            (key for key, value in encoding_strategies['Gender_encoded'].items() if value == gender_code),
            'Others'
        )

        if gender in gender_distribution:
            gender_distribution[gender] += 1
        else:
            gender_distribution['Others'] += 1

        # Decode and categorize work interfere
        work_interfere_code = item.get('work_interfere', None)
        work_interfere = next((key for key, value in encoding_strategies['work_interfere'].items() if value == work_interfere_code), 'Unknown')
        if work_interfere in work_interfere_levels:
            work_interfere_levels[work_interfere] += 1

        # Decode benefits and categorize with predictions
        benefits_code = item.get('benefits', None)
        benefits = next((key for key, value in encoding_strategies['benefits'].items() if value == benefits_code), "Unknown")
        prediction = item.get('prediction_result', 'No Treatment Needed')
        if benefits in benefits_vs_prediction:
            benefits_vs_prediction[benefits][prediction] += 1

        # Predictions
        prediction_counts[prediction] += 1

        # Age distribution
        try:
            age_distribution.append(int(item.get('Age', 0)))
        except ValueError:
            pass  # Skip invalid ages

    return render_template(
        'view_data.html',
        data=dynamo_data,
        keys=dynamo_data[0].keys() if dynamo_data else [],
        prediction_counts=prediction_counts,
        age_distribution=age_distribution,
        gender_distribution=gender_distribution,
        work_interfere_levels=work_interfere_levels,
        benefits_vs_prediction=benefits_vs_prediction
    )



@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Process data in the exact order used during training
    feature_order = [
        'Age', 'Gender_encoded', 'self_employed', 'family_history', 'work_interfere',
        'no_employees', 'tech_company', 'benefits', 'care_options', 'wellness_program',
        'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
        'physical_health_consequence', 'coworkers', 'supervisor', 'mental_vs_physical', 'sentiment_encoded'
    ]
    print("data coming from form-----------------------")
    print(data)
    processed_data = [
        float(data[field]) if field == 'Age' else encoding_strategies[field][data[field]]
        for field in feature_order
    ]

    # Check length
    if len(processed_data) != len(feature_order):
        return render_template('index.html', dynamic_fields=dynamic_fields, prediction="Feature count mismatch. Please check input data.")

    
    processed_data = np.array([processed_data])  # Ensure that processed_data is a 2D array
    print("before preprocessing----------------------")
    print(processed_data)
    

    age_index = feature_order.index('Age')  # Get the index of the 'Age' column in feature_order

    # Scale only the 'Age' column
    scaler = StandardScaler()

    # Apply scaling to 'Age' column
    processed_data[:, age_index] = scaler.fit_transform(processed_data[:, age_index].reshape(-1, 1)).flatten()
    
    print("after preprocessing----------------------")
    print(processed_data)
    prediction = model.predict(processed_data)
    print(prediction)
    result = "Needs Treatment" if prediction[0] == 1 else "No Treatment Needed"


    # Save the response and prediction result to DynamoDB
    save_to_postgresql(data, result)

    # Render the prediction on the same page
    dynamic_fields = {
        'benefits': ['Yes', 'No', "Don't know"],
        'care_options': ['Yes', 'No', 'Not sure'],
        'wellness_program': ['Yes', 'No', "Don't know"],
        'seek_help': ['Yes', 'No', "Don't know"],
        'anonymity': ['Yes', 'No', "Don't know"],
        'leave': ['Very easy', 'Somewhat easy', 'Somewhat difficult',  "Don't know"],
        'mental_health_consequence': ['Yes', 'No', 'Maybe'],
        'physical_health_consequence': ['Yes', 'No', 'Maybe'],
        'coworkers': ['Yes', 'No', 'Some of them'],
        'supervisor': ['Yes', 'No', 'Some of them'],
        'mental_vs_physical': ['Yes', 'No', "Don't know"]
    }
    return render_template('index.html', dynamic_fields=dynamic_fields, prediction=result)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='127.0.0.1', port=8080, debug=True)
