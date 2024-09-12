from utils import (
    read_video,
    save_video,
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters,
)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from audio_handler import addAudioFiles
import cv2
import pandas as pd
from copy import deepcopy, copy
from time import sleep
import pickle

def main():
    # Read Video
    constants.INPUT_VIDEO_PATH = 'input_videos/' + input("Enter video file name (without extension): ") + '.mp4'
    doubles = input("Singles / doubles? (1 / 2): ")
    if doubles == '2':
        print("Doubles functionality to be soon")
        exit()
    
    if doubles != '1':
        exit()
    
    rewrite = input("Need to rewrite detections? (y / n): ")
    rewrite = (1 if rewrite == 'y' else 0 if rewrite == 'n' else -1)
    if rewrite == -1:
        exit()

    video_frames = read_video(constants.INPUT_VIDEO_PATH)
    total_frames = len(video_frames)
    frame_rate = 24  # 24fps
    total_time_in_seconds = total_frames / frame_rate
    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")
    rewrite = 0

    frame_copy = video_frames[0].copy()

    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)

    if not rewrite:
        with open("tracker_stubs/player_detections.pkl", 'rb') as f:
            player_detections = pickle.load(f)

        with open("tracker_stubs/ball_detections.pkl", 'rb') as f:
            ball_detections = pickle.load(f)

        with open("tracker_stubs/court_keypoints.pkl", 'rb') as f:
            court_keypoints_list = pickle.load(f)
    else:
        player_detections = []
        ball_detections = []
        court_keypoints_list = []

    distances = []

    for i in range(len(video_frames)):
        if rewrite:
            keypoints = court_line_detector.get_keypoints(video_frames[i])
            player_detection = player_tracker.detect_frame(video_frames[i], keypoints)
            ball_detection = ball_tracker.detect_frame(video_frames[i])

            player_detections.append(player_detection)
            ball_detections.append(ball_detection)
            court_keypoints_list.append(keypoints)

        video_frames[i] = ball_tracker.draw_bboxes(video_frames[i], ball_detections[i])

        distances.append(abs(player_detections[i][1][1] - player_detections[i][2][1]) if 2 in player_detections[i] else 0)

        # Draw frame number on top left corner
        cv2.putText(video_frames[i], f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if rewrite:
            cv2.imshow("video_tracking", video_frames[i])
            # Wait for 25ms and check if 'q' is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    if rewrite:
        cv2.destroyAllWindows()
        sleep(1)

        with open("tracker_stubs/player_detections.pkl", 'wb') as f:
            pickle.dump(player_detections, f)

        with open("tracker_stubs/ball_detections.pkl", 'wb') as f:
            pickle.dump(ball_detections, f)

        with open("tracker_stubs/court_keypoints.pkl", 'wb') as f:
            pickle.dump(court_keypoints_list, f)

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    court_keypoints = court_line_detector.predict(frame_copy)

    # choose players
    # player_detections = player_tracker.choose_and_filter_players(
    #     court_keypoints, player_detections
    # )

    # MiniCourt
    mini_court = MiniCourt(frame_copy)

    # Detect ball shots
    ball_shot_frames, audio_frames = ball_tracker.get_ball_shot_frames(ball_detections, court_keypoints, distances)
    print("Ball shots: ", len(ball_shot_frames))

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = (
        mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints
        )
    )

    player_stats_data = [
        {
            "frame_num": 0,

            "player_1_number_of_shots": 0,
            "player_1_total_shot_speed": 0,
            "player_1_last_shot_speed": 0,
            "player_1_total_player_speed": 0,
            "player_1_last_player_speed": 0,
            "player_1_score": 0,

            "player_2_number_of_shots": 0,
            "player_2_total_shot_speed": 0,
            "player_2_last_shot_speed": 0,
            "player_2_total_player_speed": 0,
            "player_2_last_player_speed": 0,
            "player_2_score": 0,
        }
    ]

    point_time = 0
    audio_ind = 0
    rallies = []
    rally_display = {}
    audioList = []
    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]

        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        point_time += ball_shot_time_in_seconds

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1],
        )
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = (
            distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
        )

        # player who the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id], ball_mini_court_detections[start_frame][1]
            ),
        )

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        if (opponent_player_id in player_mini_court_detections[start_frame] and opponent_player_id in player_mini_court_detections[end_frame]):
            distance_covered_by_opponent_pixels = measure_distance(
                player_mini_court_detections[start_frame][opponent_player_id],
                player_mini_court_detections[end_frame][opponent_player_id],
            )

        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )

        speed_of_opponent = (
            distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6
        )

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats["frame_num"] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
        current_player_stats[
            f"player_{player_shot_ball}_total_shot_speed"
        ] += speed_of_ball_shot
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = (
            speed_of_ball_shot
        )

        current_player_stats[
            f"player_{opponent_player_id}_total_player_speed"
        ] += speed_of_opponent
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = (
            speed_of_opponent
        )

        if audio_frames[audio_ind] < ball_shot_frames[ball_shot_ind + 1]:
            current_player_stats[f"player_{player_shot_ball}_score"] += 1
            rallies.append(round(point_time, 3))
            rally_display[audio_frames[audio_ind]] = rallies.copy()
            audio_ind += 1
            point_time = 0
            audioList.append({'start_in': (audio_frames[audio_ind] - 200) // 24, 'audio_path': f'audios/pointplayer{player_shot_ball}.mp3'})

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({"frame_num": list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(
        frames_df, player_stats_data_df, on="frame_num", how="left"
    )
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df["player_1_average_shot_speed"] = (
        player_stats_data_df["player_1_total_shot_speed"]
        / player_stats_data_df["player_1_number_of_shots"]
    )
    player_stats_data_df["player_2_average_shot_speed"] = (
        player_stats_data_df["player_2_total_shot_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )
    player_stats_data_df["player_1_average_player_speed"] = (
        player_stats_data_df["player_1_total_player_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )
    player_stats_data_df["player_2_average_player_speed"] = (
        player_stats_data_df["player_2_total_player_speed"]
        / player_stats_data_df["player_1_number_of_shots"]
    )

    # Calculate total distance traveled by each player
    player_1_total_distance = (
        player_stats_data_df["player_1_total_player_speed"].iloc[-1]
        * total_time_in_seconds
        / 12.6
    )
    player_2_total_distance = (
        player_stats_data_df["player_2_total_player_speed"].iloc[-1]
        * total_time_in_seconds
        / 12.6
    )

    # Draw Player Stats
    video_frames = draw_player_stats(video_frames, player_stats_data_df, distances)

    rally_index = len(video_frames)
    for i, (frame, players, players_mini_court, ball_mini_court, keypoints) in enumerate(
        zip(
            video_frames,
            player_detections,
            player_mini_court_detections,
            ball_mini_court_detections,
            court_keypoints_list
        )
    ):
        if i in rally_display:
            rally_index = i
    
        # Draw bounding boxes for players
        if distances[i] > 150:
            frame = player_tracker.draw_bboxes(frame, players)

            # Draw mini court and points for players and ball
            frame = mini_court.draw_mini_court(frame)
            frame = mini_court.draw_points_on_mini_court(frame, players_mini_court)
            frame = mini_court.draw_points_on_mini_court(
                frame, ball_mini_court, color=(0, 255, 255)
            )
            
            frame = court_line_detector.draw_keypoints(frame, keypoints)

        if rally_index < len(video_frames):
            cv2.rectangle(frame, (8, 40), (127, 115), (0, 0, 0), -1)
            for i in range(len(rally_display[rally_index])):
                frame = cv2.putText(frame, f"Rally {i + 1} length: {rally_display[rally_index][i]} s", (10, 50 + i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.rectangle(frame, (8, 120), (200, 160), (0, 0, 0), -1)
        frame = cv2.putText(frame, f"Total distance, Player 1: {player_1_total_distance:.2f} m", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        frame = cv2.putText(frame, f"Total distance, Player 2: {player_2_total_distance:.2f} m", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Update the frame in the list
        video_frames[i] = frame

        # Display the frame
        cv2.imshow("video_analysis", frame)

        # Wait for 25ms and check if 'q' is pressed to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    save_video(video_frames, "output_videos/output_video.avi")
        
    addAudioFiles("output_videos/output_video.avi", audioList, output_path="output_videos/output_with_audio.avi")

if __name__ == "__main__":
    main()
