from sklearn.neighbors import NearestNeighbors
import numpy as np

# define a class  for 2D Kalman filter.
class KalmanFilter2D:
    def __init__(self, dt, process_var, measurement_var):
        """
        dt: time step (float)
        process_var: process variance (float)
        measurement_var: measurement variance (float)
        """
        # State vector: [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # Observation matrix (we can observe x, y only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Measurement noise covariance
        self.R = measurement_var * np.eye(2)

        # Process noise covariance
        q = process_var
        self.Q = q * np.array([
            [dt**4/4, 0,       dt**3/2, 0],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0],
            [0,       dt**3/2, 0,       dt**2]
        ])

        # Estimate error covariance
        self.P = np.eye(4)

    def predict(self):
        # Predict state and error covariance
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        """
        z: measurement [x, y]
        """
        z = np.array(z).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].flatten()

# Set up the tracking scheme.
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
