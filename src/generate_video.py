"""
Module for generating, saving, extracting frames from, and stitching videos using LLMs and Google GenAI.
"""

import pandas as pd
import os
import time
import cv2
from google import genai
from google.genai import types
from moviepy import VideoFileClip, concatenate_videoclips
from dotenv import load_dotenv
import toml

load_dotenv()


class VideoGenerator:
    """
    Handles video generation, saving, frame extraction, and stitching for viral tweet scripts.
    """

    def __init__(self):
        """
        Initializes VideoGenerator with configuration and GenAI client.
        """
        with open("../config/config.toml", "r") as f:
            self.config = toml.load(f)
        self.project_id = self.config['gcp']['project_id']
        self.location = self.config['gcp']['location']
        self.video_model = self.config['llm']['video_model']
        self.client = genai.Client(vertexai=True, project=self.project_id, location=self.location)

    def generate_video(self, prompt, duration=8, resolution="1080p", starting_image=None):
        """
        Generates a video using the GenAI client and given prompt.

        Args:
            prompt (str): Prompt for video generation.
            duration (int): Duration of the video in seconds.
            resolution (str): Video resolution.
            starting_image (str, optional): Path to starting image for video.

        Returns:
            operation: GenAI operation object for the video generation.
        """
        operation = self.client.models.generate_videos(
            model=self.video_model,
            prompt=prompt,
            image=types.Image.from_file(location=starting_image) if starting_image else None,
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                number_of_videos=1,
                duration_seconds=duration,
                resolution=resolution,
                person_generation="allow_adult",
                enhance_prompt=True,
                generate_audio=True,
            )
        )
        # Poll until operation is done
        while not operation.done:
            time.sleep(10)
            operation = self.client.operations.get(operation)
        return operation

    def save_video(self, operation, name: str, out_dir):
        """
        Saves the generated video to disk.

        Args:
            operation: GenAI operation object containing the video.
            name (str): Name for the saved video file.
            out_dir (str): Directory to save the video.

        Returns:
            str or None: Path to saved video or None if failed.
        """
        os.makedirs(out_dir, exist_ok=True)
        if operation.response:
            video_data = operation.response.generated_videos[0].video.video_bytes
            video_path = os.path.join(out_dir, f"{name}.mp4")
            with open(video_path, "wb") as f:
                f.write(video_data)
            print(f"✅ Video saved: {video_path}")
            return video_path
        else:
            print(f"⚠️ No response for {name}")
            return None

    def extract_last_frame(self, video_path, out_dir):
        """
        Extracts and saves the last frame of a video.

        Args:
            video_path (str): Path to the video file.
            out_dir (str): Directory to save the frame.

        Returns:
            str or None: Path to saved frame image or None if failed.
        """
        if video_path is None:
            return None
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_path = os.path.join(out_dir, os.path.basename(video_path).replace(".mp4", "_lastframe.jpg"))
            cv2.imwrite(frame_path, frame)
            print(f"🖼️ Last frame saved: {frame_path}")
            return frame_path
        else:
            print(f"⚠️ Failed to read last frame: {video_path}")
            return None

    def process_dataframe(self, df, out_video_root="../output/videos", out_frame_root="../output/frames", resume=True):
        """
        Processes a DataFrame of video scripts, generates videos and extracts last frames.

        Args:
            df (pd.DataFrame): DataFrame containing video scripts.
            out_video_root (str): Root directory for saving videos.
            out_frame_root (str): Root directory for saving frames.
            resume (bool): Whether to skip already existing videos/frames.

        Returns:
            pd.DataFrame: DataFrame with video and frame paths added.
        """
        video_paths = []
        frame_paths = []
        for idx, scenes in df["video_script"].items():
            if not isinstance(scenes, list):
                print(f"Skipping row {idx}, invalid scenes format")
                video_paths.append(None)
                frame_paths.append(None)
                continue
            row_video_dir = os.path.join(out_video_root, str(idx))
            row_frame_dir = os.path.join(out_frame_root, str(idx))
            os.makedirs(row_video_dir, exist_ok=True)
            os.makedirs(row_frame_dir, exist_ok=True)
            row_videos, row_frames = [], []
            starting_image = None
            for s_idx, scene in enumerate(scenes, start=1):
                video_name = f"scene{s_idx}"
                video_path = os.path.join(row_video_dir, f"{video_name}.mp4")
                frame_path = os.path.join(row_frame_dir, f"{video_name}_lastframe.jpg")
                # Resume logic: skip if video and frame already exist
                if resume and os.path.exists(video_path) and os.path.exists(frame_path):
                    print(f"⏩ Skipping {video_path}, already exists")
                    row_videos.append(video_path)
                    row_frames.append(frame_path)
                    starting_image = frame_path
                    continue
                operation = self.generate_video(scene, starting_image=starting_image)
                video_path = self.save_video(operation, video_name, row_video_dir)
                row_videos.append(video_path)
                frame_path = self.extract_last_frame(video_path, row_frame_dir)
                row_frames.append(frame_path)
                starting_image = frame_path
            video_paths.append(row_videos)
            frame_paths.append(row_frames)
        df["video_paths"] = video_paths
        df["last_frame_paths"] = frame_paths
        return df

    def stitch_videos(self, video_list, output_path):
        """
        Stitches a list of video files into a single video.

        Args:
            video_list (list): List of video file paths.
            output_path (str): Path to save the stitched video.

        Returns:
            str or None: Path to stitched video or None if failed.
        """
        clips = []
        for v in video_list:
            if v and os.path.exists(v):
                clips.append(VideoFileClip(v))
            else:
                print(f"⚠️ Skipping missing video: {v}")
        if not clips:
            print(f"❌ No valid videos to stitch for {output_path}")
            return None
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"🎬 Stitched video saved: {output_path}")
        return output_path

    def stitch_dataframe_videos(self, df, out_dir="../output/stitched_videos"):
        """
        Stitches videos for each row in the DataFrame and saves them.

        Args:
            df (pd.DataFrame): DataFrame containing video paths.
            out_dir (str): Directory to save stitched videos.

        Returns:
            pd.DataFrame: DataFrame with stitched video paths added.
        """
        os.makedirs(out_dir, exist_ok=True)
        stitched_paths = []
        for idx, row in df.iterrows():
            video_list = row.get("video_paths", [])
            tweet_dir = os.path.join(out_dir, f"tweet{idx}")
            os.makedirs(tweet_dir, exist_ok=True)
            stitched_path = os.path.join(tweet_dir, f"tweet{idx}_final.mp4")
            stitched = self.stitch_videos(video_list, stitched_path)
            stitched_paths.append(stitched)
        df["stitched_video"] = stitched_paths
        return df

    def run(self, in_csv='../data/viral_tweets_with_scripts.csv', out_csv='../data/viral_tweets_with_videos.csv'):
        """
        Runs the full pipeline: processes DataFrame, generates videos, stitches them, and saves results.

        Args:
            in_csv (str): Input CSV file path.
            out_csv (str): Output CSV file path.

        Returns:
            pd.DataFrame: DataFrame with video and frame paths.
        """
        df = pd.read_csv(in_csv)
        df = self.process_dataframe(df)
        df = self.stitch_dataframe_videos(df)
        df.to_csv(out_csv, index=False)
        return df
