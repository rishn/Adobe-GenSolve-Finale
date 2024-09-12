from ultralytics import YOLO
import cv2
import pickle
import sys

sys.path.append("../")
from utils import measure_distance, get_center_of_bbox


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frames = {}
        for i in range(10):
            player_detections_first_frames.update(player_detections[i])

        chosen_player = self.choose_players(
            court_keypoints, player_detections_first_frames
        )
        filtered_player_detections = []

        if chosen_player != [1, 2]:
            new_id = ({1, 2} - set(chosen_player)).pop()

        for player_dict in player_detections:
            filtered_player_dict = {
                (track_id if track_id < 3 else new_id): bbox
                for track_id, bbox in player_dict.items()
                if track_id in chosen_player
            }
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    def detect_frames(self, frames, stub_path, retrain=False):
        player_detections = []

        if not retrain:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if retrain:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_distance = min(
                abs(player_center[1] - court_keypoints[1]),
                abs(court_keypoints[5] - player_center[1]),
            )
            if (
                player_center[0] < court_keypoints[0]
                or player_center[0] > court_keypoints[4]
            ):
                min_distance = float("inf")
            distances.append((track_id, min_distance))

        # sort the distances in ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = []
        for i in range(min(2, len(distances))):
            chosen_players.append(distances[i][0])

        return chosen_players

    def detect_frame(self, frame, keypoints):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id == None:
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        chosen_player = self.choose_players(keypoints, player_dict)

        filtered_player_dict = {}
        for i in range(len(chosen_player)):
            filtered_player_dict[i + 1] = player_dict[chosen_player[i]]

        # filtered_player_dict = {(track_id if track_id < 3 else new_id): bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}

        return filtered_player_dict

    def draw_bboxes(self, frame, player_detections):
        # Draw Bounding Boxes
        for track_id, bbox in player_detections.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(
                frame,
                f"Player ID: {track_id}",
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        return frame
