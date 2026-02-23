import pandas as pd
from joblib import load

def predict_fitness_and_recommend(activity,age,bloodPressure,gender,heartRate,height,nutrition,sleepHours,smoke,weight):

    activity=int(activity)
    age=int(age)
    bloodPressure=int(bloodPressure)
    heartRate=int(heartRate)
    height=float(height)
    weight=float(weight)
    nutrition=int(nutrition)
    sleepHours=float(sleepHours)


    df=pd.DataFrame([{
        "age":age,
        "heart_rate":heartRate,
        "blood_pressure":bloodPressure,
        "sleep_hours":sleepHours,
        "nutrition_quality":nutrition,
        "activity_index":activity,
        "smokes":True if smoke=="Yes" else False,
        "bmi":float(weight/height**2),
        "gender_M":True if gender=="Male" else False
    }])

   

    linear_model=load("predictor.pkl")
    scaler=load("scaler.pkl")
    numeric_cols=["age","heart_rate","blood_pressure","sleep_hours","nutrition_quality","activity_index","bmi"]


    fitness=fitness_system(linear_model,scaler,numeric_cols,df.copy())
    recommendation=recommendation_system(linear_model,scaler,df.copy(),fitness,numeric_cols)

    return fitness,recommendation


def fitness_system(model,scaler,numeric_cols,df):

    df[numeric_cols]=scaler.transform(df[numeric_cols])
    fitness=round(model.predict_proba(df)[:,1][0],2)
    return fitness


def recommendation_system(model, scaler, df, fitness, numeric_cols):
    """
    Provides daily recommendations for improving fitness based on sleep, nutrition, and activity.
    """

    # Extract current values
    old_sleep = df["sleep_hours"].iloc[0]
    old_nutrition = df["nutrition_quality"].iloc[0]
    old_activity = df["activity_index"].iloc[0]

    # --- Gradual sleep adjustment ---
    sleep_step = 0.5  # max 30 min/day
    if old_sleep < 8:
        new_sleep = min(old_sleep + sleep_step, 8)
        sleep_recommendation = f"Gradually increase sleep from {old_sleep}h to {new_sleep}h today."
    elif old_sleep > 10:
        new_sleep = max(old_sleep - sleep_step, 10)
        sleep_recommendation = f"Gradually reduce sleep from {old_sleep}h to {new_sleep}h today."
    else:
        new_sleep = old_sleep
        sleep_recommendation = "Your sleep duration is in the healthy range."

    # --- Nutrition recommendation ---
    nutrition_threshold = 7  # scale 0-10
    nutrition_step = 1
    if old_nutrition < nutrition_threshold:
        new_nutrition = min(old_nutrition + nutrition_step, 10)
        nutrition_recommendation = f"Increase nutrition quality from {old_nutrition} to {new_nutrition}."
    else:
        new_nutrition = old_nutrition
        nutrition_recommendation = "Your nutrition quality is good."

    # --- Activity recommendation ---
    activity_threshold = 3  # scale 0-5
    activity_step = 0.5
    if old_activity < activity_threshold:
        new_activity = min(old_activity + activity_step, 5)
        activity_recommendation = f"Increase activity level from {old_activity} to {new_activity}."
    else:
        new_activity = old_activity
        activity_recommendation = "Your activity level is good."

    # --- Optionally check overall fitness ---
    if fitness >= 0.8:
        overall = "You are fit! Keep maintaining your lifestyle."
    else:
        overall = "You can improve your fitness by following the recommendations below."

    # Return all recommendations
    return {
        "overall": overall,
        "sleep": sleep_recommendation,
        "nutrition": nutrition_recommendation,
        "activity": activity_recommendation
    }