from sklearn.neighbors import NearestNeighbors
import numpy as np

class Tracker:
    def __init__(self, max_lost=5, iou_threshold=30):
        self.tracks = {}  # track_id -> dict with kalman, hit count, last position
        self.next_id = 0
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def update(self, detections, dt=1.0):
        updated_ids = set()

        # Predict all current tracks
        predicted_positions = []
        track_ids = []
        for tid, track in self.tracks.items():
            pred = track['kf'].predict()
            predicted_positions.append(pred)
            track_ids.append(tid)
        predicted_positions = np.array(predicted_positions)

        # Match detections to predictions using KNN
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(track_ids)

        if len(predicted_positions) > 0 and len(detections) > 0:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(predicted_positions)
            distances, indices = nbrs.kneighbors(detections)
            for det_idx, (dist, track_idx) in enumerate(zip(distances.flatten(), indices.flatten())):
                if dist < self.iou_threshold:
                    tid = track_ids[track_idx]
                    self.tracks[tid]['kf'].update(detections[det_idx])
                    self.tracks[tid]['lost'] = 0
                    self.tracks[tid]['trace'].append(detections[det_idx])
                    updated_ids.add(tid)
                    unmatched_detections.discard(det_idx)
                    unmatched_tracks.discard(tid)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            kf = KalmanFilter2D(dt=dt, process_var=1e-2, measurement_var=1.0)
            kf.update(detections[det_idx])
            self.tracks[self.next_id] = {
                'kf': kf,
                'lost': 0,
                'trace': [detections[det_idx]]
            }
            self.next_id += 1

        # Mark unmatched tracks as lost
        for tid in unmatched_tracks:
            self.tracks[tid]['lost'] += 1

        # Remove lost tracks
        to_delete = [tid for tid, t in self.tracks.items() if t['lost'] > self.max_lost]
        for tid in to_delete:
            del self.tracks[tid]

        return self.tracks
