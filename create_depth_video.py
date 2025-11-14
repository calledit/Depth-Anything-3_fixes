import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
import os
from depth_anything_3.api import DepthAnything3
from depth_scaler import EMAMinMaxScaler

def process_video(video_input, video_output, model, process_res, batch_size=1, progress_callback=None, stop_event=None, scaler=None):
    """
    Processes a single video file to create a depth map video.
    The model is passed as an argument to avoid reloading it for each video.
    """
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_input}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, fps, (frame_width, frame_height), isColor=True)

    frames_buffer = []
    
    def write_frames(frames):
        for frame in frames:
            grayscale_depth = (frame.cpu().numpy() * 255).astype(np.uint8)
            output_frame = cv2.cvtColor(grayscale_depth, cv2.COLOR_GRAY2BGR)
            output_frame = cv2.resize(output_frame, (frame_width, frame_height))
            out.write(output_frame)

    try:
        while True:
            if stop_event and stop_event.is_set():
                print("Processing stopped.")
                break
            
            ret, frame = cap.read()
            if ret:
                frames_buffer.append(frame)

            if len(frames_buffer) == batch_size or (not ret and len(frames_buffer) > 0):
                frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_buffer]

                prediction = model.inference(
                    frames_rgb,
                    process_res=process_res
                )
                
                for i in range(len(frames_buffer)):
                    depth_map = prediction.depth[i]
                    if scaler:
                        frame = scaler(torch.from_numpy(depth_map).to(model.device))
                        if frame is not None:
                            write_frames([frame])
                    else:
                        depth_map = torch.from_numpy(depth_map)
                        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                        write_frames([depth_map])

                if progress_callback:
                    progress_callback(len(frames_buffer))
                
                frames_buffer = []

            if not ret:
                break
    finally:
        if scaler:
            write_frames(scaler.flush())
        cap.release()
        out.release()

def main():
    parser = argparse.ArgumentParser(description="Create a grayscale depth map video from a video file.")
    parser.add_argument("video_input", help="Path to the input video file.")
    parser.add_argument("video_output", help="Path to the output video file.")
    parser.add_argument("--model", default="depth-anything/DA3MONO-LARGE", help="The model to use for depth estimation.")
    parser.add_argument("--process-res", type=int, default=504, help="Processing resolution for the model.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of frames to process at once. A larger batch size may increase stability but also memory usage.")
    parser.add_argument("--decay", type=float, default=0.9, help="Decay for EMA filter.")
    parser.add_argument("--buffer-size", type=int, default=22, help="Buffer size for EMA filter.")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    model = DepthAnything3.from_pretrained(args.model).to(device).eval()
    
    scaler = EMAMinMaxScaler(decay=args.decay, buffer_size=args.buffer_size)

    try:
        cap = cv2.VideoCapture(args.video_input)
        if not cap.isOpened():
            raise IOError(f"Could not open video file {args.video_input}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        with tqdm(total=total_frames, desc=f"Processing {os.path.basename(args.video_input)}") as pbar:
            def progress_update(num_processed):
                pbar.update(num_processed)

            process_video(args.video_input, args.video_output, model, args.process_res, batch_size=args.batch_size, progress_callback=progress_update, scaler=scaler)
        
        print(f"Grayscale depth video saved to {args.video_output}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()