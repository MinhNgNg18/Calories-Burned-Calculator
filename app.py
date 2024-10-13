import streamlit as st
import pickle
import pandas as pd


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


def get_input():
    df = pd.read_csv('gym_members_exercise_tracking.csv')
    st.title('Calories Burned Calculator')
    st.write("### Please give some information so that the calories burned through your exercise can be calculated:")

    age = st.number_input('Enter your age:', min_value=0, max_value=120, value=25)
    genders = ['Male', 'Female']
    work_out_types = ['Yoga', 'HIIT', 'Cardio', 'Strength']
    gender = st.selectbox('Gender', genders)
    work_out_type = st.selectbox('Work Out Type', work_out_types)

    slider_labels = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
       'Session_Duration (hours)', 'Fat_Percentage', 'Water_Intake (liters)',
       'Workout_Frequency (days/week)', 'Experience_Level']
    input_dict = {}
    for label in slider_labels:
        if label in ['Workout_Frequency (days/week)', 'Experience_Level']:
            input_dict[label] = float(st.sidebar.slider(
                label=label,
                min_value=float(df[label].min()),
                max_value=float(df[label].max()),
                value=float(df[label].median()),
                step=1.0
            ))
        elif label == 'Session_Duration (hours)':
            input_dict[label] = st.sidebar.slider(
                label='Session Duration (minutes)',
                min_value=float(df[label].min())*60,
                max_value=float(df[label].max())*60,
                value=float(df[label].mean()*60)
            )/60
        else:
            input_dict[label] = st.sidebar.slider(
                label=label,
                min_value=float(df[label].min()),
                max_value=float(df[label].max()),
                value=float(df[label].mean())
            )
    input_dict['BMI'] = input_dict['Weight (kg)']/(input_dict['Height (m)']**2)
    return age, gender, work_out_type, input_dict


def get_prediction(age, gender, work_out_type, input_dict):
    data = load_model()
    model = data['model']
    scaler = data['scaler']
    one_hot = data['one_hot']

    x = pd.DataFrame({'Age': [age], 'Gender': [gender], 'Workout_Type': [work_out_type]})
    for key, value in input_dict.items():
        x[key] = value
    cols_to_encode = ['Gender', 'Workout_Type']
    encoded_cols = pd.DataFrame(one_hot.transform(x[cols_to_encode]),
                                columns=one_hot.get_feature_names_out(cols_to_encode),
                                index=x.index)
    x = pd.concat([x.drop(cols_to_encode, axis=1), encoded_cols],
                  axis=1)
    x = scaler.transform(x)
    return model.predict(x)


def show_page():
    age, gender, work_out_type, input_dict = get_input()
    ok = st.button('Calculate Calories Burned')
    if ok:
        cal = get_prediction(age, gender, work_out_type, input_dict)
        st.subheader(f"The calories burned is: {cal[0]:.2f}")


show_page()




