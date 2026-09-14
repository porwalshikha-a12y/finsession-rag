from datetime import datetime

import pandas as pd
import streamlit as st

st.write("Hello, this is a simple Streamlit app")
st.title("User Information Form")
form_values = {
    "name": None,
    "age": None,
    "height": None,
    "dob": None

}

min_date = datetime(1970, 1, 1)  # noqa: DTZ001
max_date = datetime.now()  # noqa: DTZ005

with st.form(key='user_form'):
    form_values["name"] = st.text_input("Enter your name:", key='name')
    form_values["age"] = st.number_input("Enter your age:", min_value=0,
                                         max_value=120, key='age')
    form_values["height"] = st.number_input("Enter your height (in cm):", min_value=0.0,
                                            max_value=300.0, key='height')
    form_values["dob"] = st.date_input(
        "Enter your date of birth:", min_value=min_date, max_value=max_date)

    # This will print to the console, not the Streamlit app)

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if not all(form_values.values()):
        st.warning("Please fill in all fields before submitting.")
    else:
        st.balloons()
        st.write("Form submitted successfully!")
