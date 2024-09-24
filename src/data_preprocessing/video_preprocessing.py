"""
video_preprocessing.py

A GUI application for preprocessing videos to extract frames containing human faces or bodies.
It leverages OpenCV for video processing and Tkinter for the graphical interface.

Author: Matthew Guo
Date: 24.09.09
"""

import cv2
import os
import pandas as pd
import threading
import time
import tkinter as tk
from tkinter import ttk

class VideoProcessorGUI:
    def __init__(self, root):
        """Initialize the GUI components."""
        self.root = root
        self.root.title("Video Preprocessing")

        # Create and place widgets
        self.create_widgets()

        # Data to track processing
        self.total_frames = 0
        self.kept_frames = 0
        self.start_time = time.time()
        self.frames_processed = 0

    def create_widgets(self):
        """Create GUI widgets for the application."""
        # Label for estimated time
        self.estimate_label = tk.Label(self.root, text="Estimated Time: N/A")
        self.estimate_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Table for processed video information
        self.info_table = ttk.Treeview(self.root, columns=("video", "total_frames", "kept_frames", "time_spent"), show='headings')
        self.info_table.heading("video", text="Video")
        self.info_table.heading("total_frames", text="Total Frames")
        self.info_table.heading("kept_frames", text="Kept Frames")
        self.info_table.heading("time_spent", text="Time Spent")
        self.info_table.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        
        # Configure row and column weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.grid(row=2, column=0, padx=10, pady=10)

    def update_estimated_time(self, frame_count, processed_count):
        """Update the estimated time label based on frames processed."""
        elapsed_time = time.time() - self.start_time
        if processed_count > 0:
            avg_time_per_frame = elapsed_time / processed_count
            remaining_frames = frame_count - processed_count
            estimated_time = avg_time_per_frame * remaining_frames
            self.estimate_label.config(text=f"Estimated Time: {self.format_time(estimated_time)}")
    
    def update_table(self, video_name, total_frames, kept_frames, time_spent):
        """Update the information table with new video processing results."""
        self.info_table.insert("", "end", values=(video_name, total_frames, kept_frames, self.format_time(time_spent)))

    def update_progress(self, value, max_value):
        """Update the progress bar based on the current processing status."""
        self.progress_bar['value'] = value
        self.progress_bar['maximum'] = max_value
        self.root.update_idletasks()

    @staticmethod
    def format_time(seconds):
        """Format seconds into a more readable string format (minutes and seconds)."""
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"

def preprocess_videos(csv_file, output_folder, 
                      face_cascade_path='/haarcascade_frontalface_default.xml', 
                      fullbody_cascade_path='/haarcascade_fullbody.xml', 
                      lowerbody_cascade_path='/haarcascade_lowerbody.xml'):
    """
    Preprocess videos listed in a CSV file by extracting frames that contain human bodies or faces.

    Parameters:
    - csv_file: Path to the CSV file containing video paths.
    - output_folder: Directory where the frames will be saved.
    - face_cascade_path: Path to the Haar Cascade XML file for face detection.
    - fullbody_cascade_path: Path to the Haar Cascade XML file for full body detection.
    - lowerbody_cascade_path: Path to the Haar Cascade XML file for lower body detection.
    """
    # Initialize GUI
    root = tk.Tk()
    gui = VideoProcessorGUI(root)
    root.update()

    # Load the cascade classifiers for face, full body, and lower body detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
    fullbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + fullbody_cascade_path)
    lowerbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + lowerbody_cascade_path)
    
    # Read the CSV file containing video paths
    data = pd.read_csv(csv_file)
    
    if 'Video Path' not in data.columns:
        raise ValueError("CSV file must contain a 'Video Path' column.")

    # Process each video in a separate thread
    def process_videos():
        total_videos = len(data)
        for idx, video_path in enumerate(data['Video Path']):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_folder = os.path.join(output_folder, video_name)
            
            # Create the output folder if it doesn't exist
            os.makedirs(video_output_folder, exist_ok=True)
            
            # Process the video
            total_frames, kept_frames, time_spent = _preprocess_video(video_path, video_output_folder, 
                                                                      face_cascade, fullbody_cascade, lowerbody_cascade, gui)
            
            # Update the table with processed video info
            gui.update_table(video_name, total_frames, kept_frames, time_spent)
            root.update_idletasks()  # Update GUI

            # Update progress bar
            gui.update_progress(idx + 1, total_videos)
    
    # Start the video processing in a separate thread
    processing_thread = threading.Thread(target=process_videos)
    processing_thread.start()
    root.mainloop()

def _preprocess_video(video_path, output_folder, face_cascade, fullbody_cascade, lowerbody_cascade, gui):
    """
    Process a single video to extract frames containing faces or human bodies.

    Parameters:
    - video_path: Path to the video file.
    - output_folder: Directory where the frames will be saved.
    - face_cascade: The Haar Cascade classifier for face detection.
    - fullbody_cascade: The Haar Cascade classifier for full body detection.
    - lowerbody_cascade: The Haar Cascade classifier for lower body detection.
    """
    cap = cv2.VideoCapture(video_path)
    
    frame_index = 0
    saved_frame_index = 0
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_index += 1
        
        # Process every 16th frame
        if frame_index % 16 != 0:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(60 , 60))
        fullbodies = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 90))
        lowerbodies = lowerbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        
        if len(faces) > 0 or len(fullbodies) > 0 or len(lowerbodies) > 0:
            output_path = os.path.join(output_folder, f"frame_{frame_index}.jpg")
            cv2.imwrite(output_path, frame)
            saved_frame_index += 1
        
        # Update GUI with estimated time
        elapsed_time = time.time() - start_time
        gui.update_estimated_time(total_frame_count, frame_index)
    
    cap.release()
    cv2.destroyAllWindows()
    
    time_spent = time.time() - start_time
    return total_frame_count, saved_frame_index, time_spent
