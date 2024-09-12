import numpy as np
import cv2

def draw_player_stats(output_video_frames, player_stats, distances):

    for index, row in player_stats.iterrows():        
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']
        player_1_score = row['player_1_score']
        player_2_score = row['player_2_score']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        frame = output_video_frames[index]

        # Scale down width and height by 50%
        width = 350 // 2
        height = 230 // 2

        # Scale down position by 50%
        start_x = frame.shape[1] - 400 // 2
        start_y = frame.shape[0] - 210 // 2
        end_x = start_x + width
        end_y = start_y + height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        if distances[index] > 150:
            # Scale down font size and adjust text positions accordingly
            text = "     Player 1     Player 2"
            frame = cv2.putText(frame, text, (start_x + 40, start_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            text = "Shot Speed"
            frame = cv2.putText(frame, text, (start_x + 5, start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
            frame = cv2.putText(frame, text, (start_x + 65, start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            text = "Player Speed"
            frame = cv2.putText(frame, text, (start_x + 5, start_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
            frame = cv2.putText(frame, text, (start_x + 65, start_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            text = "avg. S. Speed"
            frame = cv2.putText(frame, text, (start_x + 5, start_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
            frame = cv2.putText(frame, text, (start_x + 65, start_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            text = "avg. P. Speed"
            frame = cv2.putText(frame, text, (start_x + 5, start_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
            frame = cv2.putText(frame, text, (start_x + 65, start_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            text = "score"
            frame = cv2.putText(frame, text, (start_x + 5, start_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            text = f"0/0/{player_1_score}    0/0/{player_2_score}"
            frame = cv2.putText(frame, text, (start_x + 65, start_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return output_video_frames
