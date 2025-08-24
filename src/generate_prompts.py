"""
Module for generating video scripts from viral tweets using LLMs and prompt templates.
"""

import pandas as pd
import toml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Scene(BaseModel):
    """
    Represents a single scene in a storyboard.
    """
    index: int = Field(..., description="Scene number from 1 to 8")
    description: str = Field(..., description="Compelling, cinematic, viral-potential scene description")


class Storyboard(BaseModel):
    """
    Represents a storyboard consisting of multiple scenes.
    """
    scenes: List[Scene] = Field(..., description="List of 8 sequential scenes that form a viral short story")


class PromptGenerator:
    """
    Generates video scripts from viral tweets using prompt templates and LLMs.
    """

    def __init__(self, config_path="../config/config.toml"):
        """
        Initializes PromptGenerator with configuration and LLM models.

        Args:
            config_path (str): Path to the configuration TOML file.
        """
        with open(config_path, "r") as f:
            self.config = toml.load(f)
        # Initialize LLMs for different tasks
        self.scene_llm = ChatOpenAI(model=self.config['llm']['scene_model'], temperature=0.5)
        self.script_llm = ChatOpenAI(model=self.config['llm']['script_model'], temperature=0.7)
        self.extraction_llm = ChatOpenAI(model=self.config['llm']['extraction_model'], temperature=0)
        # Prompt templates for scene generation and script writing
        self.scene_generation_prompt = PromptTemplate(
            input_variables=["tweet"],
            template=self.config['prompts']['scene_generation_prompt']
        )
        self.tweet_to_multiscene_prompt = PromptTemplate(
            input_variables=["tweet_text", "scene_number", "scene_description", "previous_context"],
            template=self.config['prompts']['tweets_to_multiscene_prompt']
        )
        # Output parser for extracting structured JSON
        self.output_parser = JsonOutputParser(pydantic_object=Storyboard)
        self.json_extraction_prompt_tempelate = PromptTemplate(
            input_variables=["text"],
            template=self.config['prompts']['json_extraction_prompt'],
            partial_variables={"format_instruction": self.output_parser.get_format_instructions()}
        )

    def generate_video_script(self, tweet):
        """
        Generates a video script from a tweet by creating scenes and writing scripts for each scene.

        Args:
            tweet (str): The tweet text to generate a video script from.

        Returns:
            list: List of script outputs for each scene.
        """
        # Generate scenes from the tweet
        scene_chain = self.scene_generation_prompt | self.scene_llm | StrOutputParser()
        json_chain = self.json_extraction_prompt_tempelate | self.extraction_llm | self.output_parser
        scenes_json = scene_chain.invoke({"tweet": tweet})
        scenes_json = json_chain.invoke({"text": scenes_json})
        scene_outputs = []
        previous_context = "Start of video"
        script_chain = self.tweet_to_multiscene_prompt | self.script_llm | StrOutputParser()
        # For each scene, generate the script using the LLM
        for scene in scenes_json["scenes"]:
            scene_number = scene["index"]
            scene_description = scene["description"]
            result = script_chain.invoke({
                "scene_number": scene_number,
                "scene_description": scene_description,
                "previous_context": previous_context
            })
            scene_outputs.append(result)
            previous_context += f" Summary of Scene {scene_number}: {scene_description}..."
        return scene_outputs

    def run(self, in_csv='../data/viral_tweets_classified.csv', out_csv='../data/viral_tweets_with_scripts.csv'):
        """
        Runs the full pipeline: reads classified tweets, generates video scripts, and saves to CSV.

        Args:
            in_csv (str): Input CSV file path containing classified tweets.
            out_csv (str): Output CSV file path for tweets with video scripts.

        Returns:
            pd.DataFrame: DataFrame with video scripts added.
        """
        df = pd.read_csv(in_csv)
        # Filter for meaningful tweets only
        df_meaningful = df[df['is_meaningful'] == 'YES'].copy()
        # Generate video script for each tweet
        df_meaningful['video_script'] = df_meaningful['text'].apply(self.generate_video_script)
        df_meaningful.to_csv(out_csv, index=False)
        return df_meaningful