import argparse
import requests
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import json
import pytz

def load_config(filename="config.json"):
    """Load configuration from a JSON file."""
    with open(filename, "r") as f:
        config = json.load(f)
    return config

config = load_config()

API_ENDPOINT = config["API_ENDPOINT"]

def api_request(url, params=None):
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    return response

def get_all_sports(api_key):
    params = {'api_key': api_key, 'all': 'true'}
    sports_data = api_request(API_ENDPOINT, params)
    return pd.DataFrame(sports_data.json())


def get_active_sport_keys(api_key: str, sport_list: List[str]) -> Optional[List[str]]:
    """
    Retrieve a list of active sports keys based on a provided list of sports.

    This function fetches information about all available sports from an API using the given API key.
    It then filters the results to return only the sports that are both active and present in the provided list of sports.
    """
    sport_df = get_all_sports(api_key)
    
    if sport_df is None:
        print("Failed to fetch sports data.")
        return None

    active_sports = sport_df[sport_df['key'].isin(sport_list) & sport_df['active']]['key']
    
    return active_sports.tolist()

def fetch_odds_data(sport, api_key, regions, markets, odds_format, date_format):
    """
    Fetch odds data from the API.
    
    Returns:
    - JSON data if successful.
    - None if request fails.
    """
    url = f'{API_ENDPOINT}/{sport}/odds'
    params = {
        'api_key': api_key,
        'regions': regions,
        'markets': markets,
        'oddsFormat': odds_format,
        'dateFormat': date_format,
    }
    response = api_request(url, params=params)

    print('Remaining requests', response.headers['x-requests-remaining'])
    return response.json()

def parse_odds_data_to_df(odds_json):
    """
    Parse the provided odds JSON data into a pandas DataFrame and filtering for the games
    within 27 hours from now
    """
    rows = []
    
    for match in odds_json:
        match_id = match['id']
        sport_title = match['sport_title']
        match_date = match['commence_time']
        home_team = match['home_team']
        away_team = match['away_team']

        for bookmaker in match['bookmakers']:
            bookmaker_name = bookmaker['title']
            
            for market in bookmaker['markets']:
                market_name = market['key']
                home_odds = next((outcome['price'] for outcome in market['outcomes'] if outcome['name'] == home_team), None)
                away_odds = next((outcome['price'] for outcome in market['outcomes'] if outcome['name'] == away_team), None)
                draw_odds = next((outcome['price'] for outcome in market['outcomes'] if outcome['name'] == 'Draw'), None)

                rows.append([match_id, sport_title, match_date, home_team, away_team, bookmaker_name, market_name, home_odds, away_odds, draw_odds])
    
    df = pd.DataFrame(rows, columns=['Match ID', 'Sport Title', 'Match Date & Time', 'Home Team', 'Away Team', 'Bookmaker', 'Market', 'Home Team Odds', 'Away Team Odds', 'Draw Odds'])

    # Specify the data types for each column
    data_types = {
        'Match ID': 'object',
        'Sport Title': 'object',
        'Match Date & Time': 'datetime64[ns, UTC]',
        'Home Team': 'object',
        'Away Team': 'object',
        'Bookmaker': 'object',
        'Market': 'object',
        'Home Team Odds': 'float64',
        'Away Team Odds': 'float64',
        'Draw Odds': 'float64'
    }

    # Convert columns to their respective data types
    df = df.astype(data_types)
    # Current Timestamp for "current time - 24 hours"
    current_utc_timestamp = datetime.datetime.now(pytz.utc)
    timestamp_27_hours_ago = current_utc_timestamp + datetime.timedelta(hours=27)
    # Filter rows in DataFrame based on the timestamp
    filtered_last_27_hours = df[df['Match Date & Time'] <= timestamp_27_hours_ago]

    return filtered_last_27_hours

def fetch_and_parse_multiple_sport_odds_data(sports, api_key, regions, markets, odds_format, date_format):
    """
    Fetch and parse odds data for multiple sports from an API.

    This function iterates over a list of sports, fetches odds data for each sport 
    using the provided API key and parameters, parses the fetched data into a pandas DataFrame,
    and then concatenates these DataFrames into a final combined DataFrame.
    """
    dfs = []  # List to store DataFrames for each sport
    for sport in sports:
        odds_json = fetch_odds_data(sport, api_key, regions, markets, odds_format, date_format)
        if odds_json:
            df_sport = parse_odds_data_to_df(odds_json)
            dfs.append(df_sport)

    df_sport_final = pd.concat(dfs, ignore_index=True)  # Concatenate all DataFrames in the list
    print(f'There are {df_sport_final.shape[0]} matches within 27 hours from now, from the selected leagues')

    return df_sport_final

def calculate_percentage_difference(a, b):
    """
    Calculate the percentage difference between two numbers. a being the exchange odds and b the site odds
    """
    if a == 0:
        return 0
    return (a - b) / b * 100

def calculate_profit_per_unit_staked(odds_site, odds_exchange, discount=0.15, exchange_commission=0.035):
    """
    Calculate the profit percentage between two odds given a discount rate and exchange comission
    """
    net_profit_per_unit_staked = (odds_site - 1 + (odds_site-discount) * (1-odds_exchange) / (odds_exchange-exchange_commission)) * 100
    return net_profit_per_unit_staked

def calculate_profit_risk_ratio(odds_site, odds_exchange, discount=0.15, exchange_commission=0.035):
    """
    Calculate the profit risk ratio between two odds given a discount rate and exchange comission
    """
    nominator = ((1 - exchange_commission) - (1 - discount) * (odds_exchange - exchange_commission) / (odds_site - discount))
    denominator = (odds_exchange - 1)
    profit_risk_ratio =  nominator / denominator * 100
    return profit_risk_ratio

def split_odds_data(odds_data):
    """
    Split the provided odds dataframe into exchange (betfair) lay odds
    """
    exchange = odds_data[(odds_data['Bookmaker'] == 'Betfair') & (odds_data['Market'] == 'h2h_lay')]
    site = odds_data[(odds_data['Bookmaker'] == 'Pinnacle')]
    return exchange, site

def calculate_odds_difference(exchange_df, site_df):
    """
    Compute the percentage differences in odds between Betfair's lay market 
    and other bookmakers for each match.
    """
    results = []

    # Iterate through Betfair lay odds
    
    for index, row in exchange_df.iterrows():
        match_id = row['Match ID']
        match_date_time = row['Match Date & Time']
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_team_odds_exchange = row['Home Team Odds']
        away_team_odds_exchange = row['Away Team Odds']
        draw_odds_exchange = row['Draw Odds']
        
        # Filter odds for the same match from other bookmakers
        other_bookmakers = site_df[(site_df['Match ID'] == match_id)]
        
        for _, other_row in other_bookmakers.iterrows():
            home_team_odds_site = other_row['Home Team Odds']
            away_team_odds_site = other_row['Away Team Odds']
            draw_odds_site = other_row['Draw Odds']

            results.append({
                'Match ID': match_id,
                'Match Date & Time': row['Match Date & Time'],
                'Sport Title': row['Sport Title'],
                'Bookmakers': (other_row['Bookmaker'], row['Bookmaker']),
                'Teams': (home_team, away_team),
                '1': (home_team_odds_site, home_team_odds_exchange),
                'X': (draw_odds_site, draw_odds_exchange),
                '2': (away_team_odds_site, away_team_odds_exchange),
                '1 % Difference': calculate_percentage_difference(home_team_odds_exchange, home_team_odds_site),
                'X % Difference': calculate_percentage_difference(draw_odds_exchange, draw_odds_site),
                '2 % Difference': calculate_percentage_difference(away_team_odds_exchange, away_team_odds_site),
                '1 % Profit': calculate_profit_per_unit_staked(home_team_odds_site, home_team_odds_exchange),
                'X % Profit': calculate_profit_per_unit_staked(draw_odds_site, draw_odds_exchange),
                '2 % Profit': calculate_profit_per_unit_staked(away_team_odds_site, away_team_odds_exchange),
                '1 % Risk': calculate_profit_risk_ratio(home_team_odds_site, home_team_odds_exchange),
                'X % Risk': calculate_profit_risk_ratio(draw_odds_site, draw_odds_exchange),
                '2 % Risk': calculate_profit_risk_ratio(away_team_odds_site, away_team_odds_exchange)                  
            })

    return pd.DataFrame(results)

def get_odds_comparison(odds_data):
    """
    Compare odds between Betfair's lay market and another bookmaker (Pinnacle) for each match.
    
    This function splits the input odds data into two parts: one for the Betfair lay market 
    and another for the Pinnacle bookmaker. It then computes the percentage differences 
    in odds between these two bookmakers for each match and returns the results 
    in a structured DataFrame.
    """
    exchange_df, site_df = split_odds_data(odds_data)
    final_df = calculate_odds_difference(exchange_df, site_df)
    return final_df

def extract_opportunities(df, difference_threshold=3, profit_threshold=1):
    """
    Extract betting opportunities from a DataFrame based on set conditions.
    """
    # Define opportunities
    opportunities = []
    
    # Define the types and their corresponding columns
    types = {
        'Home Team': ('1', '1 % Difference', '1 % Profit', '1 % Risk'),
        'Draw': ('X', 'X % Difference', 'X % Profit', 'X % Risk'),
        'Away Team': ('2', '2 % Difference', '2 % Profit', '2 % Risk')
    }

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        for opp_type, (odds_col, diff_col, profit_col, risk_col) in types.items():
            if row[diff_col] < difference_threshold and row[profit_col] > profit_threshold:
                opportunities.append({
                    'Match Date & Time': row['Match Date & Time'],
                    'Sport Title': row['Sport Title'],
                    'Bookmakers': row['Bookmakers'],
                    'Teams': row['Teams'],
                    'Option': opp_type,
                    'Back, Lay': row[odds_col],
                    '% Profit': round(row[profit_col],2),
                    '% Difference': round(row[diff_col],2),
                    '% Risk': round(row[risk_col],2)
                })

    # Convert opportunities list to a DataFrame
    df_opportunities = pd.DataFrame(opportunities)
    
    # Check if the DataFrame is empty
    if df_opportunities.empty:
        print("No opportunities found in the DataFrame.")
        return None  # or return an empty DataFrame: return pd.DataFrame()

    # Return the DataFrame sorted by '% Risk' in descending order
    return df_opportunities.sort_values(by='% Risk', ascending=False)