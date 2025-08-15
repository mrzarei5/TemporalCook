from moviepy import *
import whisper
import numpy as np
import os
import sys


# Suppress output
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class TranscriptExtractor:
    #class AnswerInference(BaseModel):
    #    question_is_discussed: Literal["Yes","No"] = Field(...,description="state whether the question has been discussed in the procedure steps of the video.")
    #    answer_inferable: Literal["Yes","No"] = Field(...,description="state whether the ground truth answer can be extracted from the given procedure steps.")
    #    IDs: List[int] = Field(default_factory=list, description="state the ID(s) of the procedure steps that help extracting the ground truth answer. Empty if the ground truth answer is not inferable.") 
    
    def __init__(self, model = "base", audio_file_temp = 'temp_audio.wav'):

        self.model = whisper.load_model(model)
        self.model.to("cuda:0")
        self.audio_file_temp = audio_file_temp

        
    
    def transcribe_audio(self, audio_path):
    
        result = self.model.transcribe(audio_path)
        return result["text"]


    
    def extract_audio(self,video_path, audio_path=None):
        """
        Extracts audio from a video file and saves it as an audio file.
        
        Args:
            video_path (str): Path to the input video file.
            audio_path (str): Path to save the extracted audio file.
        """
        if not audio_path:
            audio_path = self.audio_file_temp
        
        with HiddenPrints():
            with VideoFileClip(video_path) as video:
        
                video.audio.write_audiofile(audio_path, logger=None)
        #video.close()
    
    def video_to_transcript(self,video_path):

        # Step 1: Extract audio to memory
        print("Extracting audio from video...")
        #audio_data, sample_rate = self.extract_audio_to_memory(video_path)
        self.extract_audio(video_path)
        # Step 2: Transcribe the audio directly from memory
        print("Transcribing audio...")
        transcript = self.transcribe_audio(self.audio_file_temp)
        
        return transcript

    def save_transcript(self,transcript, output_path):
        """
        Saves the transcript to a text file.
        
        Args:
            transcript (str): Transcribed text.
            output_path (str): Path to save the transcript.
        """
        with open(output_path, "w") as f:
            f.write(transcript)