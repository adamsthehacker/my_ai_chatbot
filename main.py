import streamlit as st
from bot import chat


display_text = """
<h6 style='color: blue;'>
Bot: Hello i am your ai girlfriend how can i help you
</h6>
"""

st.write("<h3 style='text-decoration: underline;'>My AI Girlfriend</h3>",
         unsafe_allow_html=True)

user_text_box = st.text_input(label='User Input',
                              help='Chat with your ai girlfriend',
                              placeholder='Type something',
                              label_visibility='collapsed',)
user_text_submit_button = st.button('Submit')

st.write(display_text, unsafe_allow_html=True)
if user_text_submit_button:
        st.markdown(f"""
    <h6 style='color: green;'>
    User: {user_text_box}
    </h6>
    """, unsafe_allow_html=True)
        st.write(f"""
    <h6 style='color: blue;'>
    Bot: {chat(user_text_box)}
    </h6>
    """, unsafe_allow_html=True)