"""
Module for fetching viral tweets using a Twitter API and classifying them using an LLM.
"""

import pandas as pd
import requests
import os
import toml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class TweetFetcherClassifier:
    """
    Fetches viral tweets based on a query and classifies them using a language model.
    """

    def __init__(self, config_path="../config/config.toml"):
        """
        Initializes the TweetFetcherClassifier with configuration and API credentials.

        Args:
            config_path (str): Path to the configuration TOML file.
        """
        load_dotenv()
        with open(config_path, "r") as f:
            self.config = toml.load(f)
        self.api_base_url = self.config['twitter']['api_base_key']
        self.api_key = os.getenv("X_API_KEY")
        self.headers = {"X-API-Key": self.api_key}
        self.classifier_prompt = self.config['prompts']['tweet_classifier_prompt']
        self.llm = ChatOpenAI(model=self.config['llm']['classifier_model'], temperature=0.5)
        self.min_likes = self.config['twitter'].get('min_likes', 100)
        self.max_results = self.config['twitter'].get('max_results', 200)
        self.page_size = self.config['twitter'].get('page_size', 50)

    def search_viral_tweets(self, query, min_likes=100, max_results=200, page_size=50):
        """
        Searches for viral tweets matching the query and minimum likes.

        Args:
            query (str): Search query.
            min_likes (int): Minimum number of likes for a tweet to be considered viral.
            max_results (int): Maximum number of tweets to fetch.
            page_size (int): Number of tweets per API request.

        Returns:
            list: List of tweet dictionaries.
        """
        full_query = f"{query} min_faves:{min_likes} lang:en"
        url = f"{self.api_base_url}/twitter/tweet/advanced_search"
        params = {
            "query": full_query,
            "max_results": page_size,
            "tweet.fields": "text,author_id,created_at,public_metrics,lang"
        }
        all_tweets = []
        next_cursor = ""
        # Paginate through results until max_results is reached or no more pages
        while len(all_tweets) < max_results:
            if next_cursor:
                params["cursor"] = next_cursor
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code} {response.text}")
            data = response.json()
            tweets = data.get("tweets", [])
            all_tweets.extend(tweets)
            next_cursor = data.get("next_cursor") or ""
            if not next_cursor:
                break
        return all_tweets[:max_results]

    def extract_viral_info(self, tweets):
        """
        Extracts relevant information from tweet objects.

        Args:
            tweets (list): List of tweet dictionaries.

        Returns:
            list: List of dictionaries with selected tweet info.
        """
        # Extract only the necessary fields from each tweet
        return [{
            "text": tweet["text"],
            "likes": tweet.get("public_metrics", {}).get("like_count", 0),
            "retweets": tweet.get("public_metrics", {}).get("retweet_count", 0),
            "replies": tweet.get("public_metrics", {}).get("reply_count", 0),
            "created_at": tweet.get("created_at"),
            "lang": tweet.get("lang")
        } for tweet in tweets]

    def classify_tweets(self, df):
        """
        Classifies each tweet in the DataFrame as meaningful or not using the LLM.

        Args:
            df (pd.DataFrame): DataFrame containing tweet information.

        Returns:
            pd.DataFrame: DataFrame with an added 'is_meaningful' column.
        """
        tweet_prompt_classifier = PromptTemplate(
            input_variables=["tweet_text"],
            template=self.classifier_prompt,
        )
        # Iterate through each tweet and classify using the LLM
        for idx, row in df.iterrows():
            classifier_chain = tweet_prompt_classifier | self.llm | StrOutputParser()
            result = classifier_chain.invoke({"tweet_text": row['text']})
            df.at[idx, 'is_meaningful'] = result
        return df

    def run(self, query, out_csv='../data/viral_tweets_classified.csv'):
        """
        Runs the full pipeline: fetches tweets, extracts info, classifies, and saves to CSV.

        Args:
            query (str): Search query for fetching tweets.
            out_csv (str): Output CSV file path.

        Returns:
            pd.DataFrame: Classified DataFrame.
        """
        tweets = self.search_viral_tweets(query, min_likes=self.min_likes,
                                          max_results=self.max_results,
                                          page_size=self.page_size)
        info = self.extract_viral_info(tweets)
        df = pd.DataFrame(info)
        df = self.classify_tweets(df)
        df.to_csv(out_csv, index=False)
        return df
