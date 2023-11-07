import streamlit as st
from helpers import *
import time
from io import StringIO

def convert_csv(df):
    return df.to_csv(index=False, encoding='utf-8')

def succes_message():
    succes = st.success('Arquivo baixado com sucesso!', icon = "✅")
    time.sleep(5)
    succes.empty()
    return

config = load_config()

API_ENDPOINT = config["API_ENDPOINT"]
DISCOUNT_RATE = config["DISCOUNT_RATE"]
EXCHANGE_COMMISSION = config["EXCHANGE_COMMISSION"]
API_KEY = config["API_KEY"]
REGIONS = config['REGIONS']
MARKETS = config['MARKETS']
ODDS_FORMAT = config['ODDS_FORMAT']
DATE_FORMAT = config['DATE_FORMAT']
SPORT_LIST = config["SPORT_LIST"]

sport_df = get_all_sports(API_KEY)

st.set_page_config(layout='wide')

st.title('BETTING POSSIBILITIES APP')

tab1, tab2, tab3 = st.tabs(['Sports list', 'Opportunities Search', 'Calculator'])

with tab1:
    st.header('Sports availables for search')

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(sport_df)
    with col2:
        sports_list = st.multiselect('Select the sports you would like to search for opportunities', sport_df.key.unique(), SPORT_LIST)

import streamlit as st
import pandas as pd
from io import StringIO

# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    if df is not None:
        # Create a string buffer
        buffer = StringIO()
        # Write the DataFrame as a CSV to the buffer
        df.to_csv(buffer, index=False)
        # Seek to the start of the string buffer
        buffer.seek(0)
        # Return the buffer value
        return buffer.getvalue()
    return None

# Function that is called when the download button is clicked
def success_message():
    st.success("Download started!")

# Initialize session state for the DataFrames
if 'df_odds' not in st.session_state:
    st.session_state['df_odds'] = None
if 'opportunitites_df' not in st.session_state:
    st.session_state['opportunitites_df'] = None

with tab2: 
    col1, col2 = st.columns(2)

    with col1:
        st.header(f'Extract match data')
        with st.expander('Sports List'):
            st.write(sports_list)
        if st.button('Extract matches within 27 hours from now'):
            st.session_state['df_odds'] = fetch_and_parse_multiple_sport_odds_data(sports_list, API_KEY, REGIONS, MARKETS, ODDS_FORMAT, DATE_FORMAT)
            if st.session_state['df_odds'] is not None:
                st.write(f'There are :blue[{len(st.session_state["df_odds"]["Match ID"].unique())}] unique matches within 27 hours from now, from the selected leagues')
        
    with col2:
        st.header(f'Extract Opportunities')
        col21, col22 = st.columns(2)
        with col21:
            difference_threshold = st.number_input('Difference between odds', 0, 100, 3)
        with col22:
            profit_threshold = st.number_input('Profit Treshold', 0, 100, 1)

        if st.button('Compare odds and extract opportunities'):
            # Check if df_odds is available before trying to process it
            if st.session_state['df_odds'] is not None:
                compared_odds = get_odds_comparison(st.session_state['df_odds'])
                st.session_state['opportunitites_df'] = extract_opportunities(compared_odds, difference_threshold=difference_threshold, profit_threshold=profit_threshold)

                if st.session_state['opportunitites_df'] is not None and not st.session_state['opportunitites_df'].empty:
                    # Display the DataFrame
                    st.dataframe(st.session_state['opportunitites_df'])
                    st.write(f':blue[{st.session_state["opportunitites_df"].shape[0]}] matches are potential bets')
                else:
                    st.error("No opportunities found with the given thresholds.")
            else:
                st.error("No odds data available. Please extract matches first.")

    # Assuming opportunitites_df should only be downloaded if it exists
    if st.session_state['opportunitites_df'] is not None and not st.session_state['opportunitites_df'].empty:
        with st.expander('Dowload data'):
            col1, col2 = st.columns(2)
            with col1:
                file_name = st.text_input('Enter file name', key='file_name')
                if not file_name.endswith('.csv'):
                    file_name += '.csv'
            with col2:
                st.download_button(
                    'Download the file in CSV format',
                    data=convert_df_to_csv(st.session_state['opportunitites_df']),
                    file_name=file_name,
                    mime='text/csv',
                    on_click=success_message
                )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.number_input('Odds Bet365')
            st.number_input('Discount')
        with col2:
            st.number_input('Odds Exchange')
            st.number_input('Taxa Exchange')