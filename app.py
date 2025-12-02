import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
from av import VideoFrame
import queue
import pandas as pd
import altair as alt
import threading
import json
import os

#saving the login info of the user
def load_users():
    """Load users from a JSON file"""
    try:
        if os.path.exists("users.json"):
            with open("users.json", "r") as f:
                return json.load(f)
        return {}
    except:
        return {}

def save_users(users):
    """Save users to a JSON file"""
    try:
        with open("users.json", "w") as f:
            json.dump(users, f)
    except:
        pass

def load_workout_data():
    """Load workout data from a JSON file"""
    try:
        if os.path.exists("workout_data.json"):
            with open("workout_data.json", "r") as f:
                return json.load(f)
        return []
    except:
        return []

def save_workout_data(data):
    """Save workout data to a JSON file"""
    try:
        with open("workout_data.json", "w") as f:
            json.dump(data, f, default=str)  # default=str handles timestamps
    except:
        pass


st.set_page_config(page_title="IskoCoach Prototype", layout="centered")

# ----------------------
# Initialize session state safely
# ----------------------
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("current_user", None)
st.session_state.setdefault("start_squats", False)
st.session_state.setdefault("target_reps", 10)
st.session_state.setdefault("workout_result_queue", queue.Queue())

if "users" not in st.session_state:
    st.session_state.users = load_users()  # Load from file

if "workout_data" not in st.session_state:
    st.session_state.workout_data = load_workout_data()  # Load from file


# ----------------------
# Login/Register
# ----------------------
def login_screen():
    st.title("🏋️ IskoCoach Login")
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.success("Logged in!")
            st.rerun()
        else:
            st.error("Invalid username/password")

    st.subheader("Register")
    new_user = st.text_input("New Username", key="reg_username")
    new_pass = st.text_input("New Password", type="password", key="reg_password")
    if st.button("Register"):
        if new_user in st.session_state.users:
            st.error("Username already exists")
        else:
            st.session_state.users[new_user] = new_pass
            save_users(st.session_state.users)  # <-- ADD THIS LINE
            st.success("Registered! You can login now")

# ----------------------
# Squat Pose Coach
# ----------------------
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def is_right_angle_at_knee(head, knee, opposite_knee):
    a = np.array(head)
    b = np.array(knee)
    c = np.array(opposite_knee)
    v1 = a - b
    v2 = c - b
    dot = np.dot(v1, v2)
    return abs(dot) < 3000  # tolerance

class PoseCoach(VideoTransformerBase):
    def __init__(self, target_reps=10, result_queue=None):
        self.pose = mp_pose.Pose()
        self.knee_suggestion = ""
        self.back_suggestion = ""
        self.knee_color = (0, 255, 0)
        self.back_color = (0, 255, 0)
        self.reps = 0
        self.squat_state = "up"
        self.score = 100
        self.target_reps = target_reps
        self.finished = False
        self.in_position = False
        self.MIN_RATIO = 0.25
        self.MAX_RATIO = 0.7
        self.result_queue = result_queue
        self._workout_saved = False
        self._lock = threading.Lock()

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        small_img = cv2.resize(img, (320, 240))
        rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_small)

        self.knee_color = (0, 255, 0)
        self.back_color = (0, 255, 0)

        if results.pose_landmarks and not self.finished:
            lm = results.pose_landmarks.landmark
            scale_x = w / 320
            scale_y = h / 240

            def to_original(p):
                return (int(p.x * 320 * scale_x), int(p.y * 240 * scale_y))

            head = to_original(lm[mp_pose.PoseLandmark.NOSE])
            left_hip = to_original(lm[mp_pose.PoseLandmark.LEFT_HIP])
            right_hip = to_original(lm[mp_pose.PoseLandmark.RIGHT_HIP])
            left_knee = to_original(lm[mp_pose.PoseLandmark.LEFT_KNEE])
            right_knee = to_original(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
            left_ankle = to_original(lm[mp_pose.PoseLandmark.LEFT_ANKLE])

            groin = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
            hip_to_ankle = abs(groin[1] - left_ankle[1])
            ratio = hip_to_ankle / h
            self.in_position = self.MIN_RATIO <= ratio <= self.MAX_RATIO

            knee_angle = calculate_angle(groin, left_knee, left_ankle)

            # Squat state and reps
            DOWN_THRESHOLD = 130
            UP_THRESHOLD = 170

            if self.in_position:
                if self.squat_state == "up" and knee_angle < DOWN_THRESHOLD:
                    self.squat_state = "down"
                elif self.squat_state == "down" and knee_angle > UP_THRESHOLD:
                    self.squat_state = "up"
                    self.reps += 1

            self.knee_suggestion = ""
            self.back_suggestion = ""

            # Check if workout is finished
            if self.reps >= self.target_reps and not self.finished:
                with self._lock:
                    if not self._workout_saved:
                        self.finished = True
                        # Put result in queue for main thread to process
                        if self.result_queue:
                            try:
                                self.result_queue.put({
                                    "reps": self.reps,
                                    "score": self.score,
                                    "finished": True
                                }, block=False)
                            except queue.Full:
                                pass
                        self._workout_saved = True

            # Suggestions after 1 rep
            if self.reps > 0 and not self.finished:
                wrong_count = 0

                # Knee suggestion
                if self.squat_state == "down":
                    if knee_angle < 90:
                        self.knee_suggestion = "⬆️ You're going too low!"
                        self.knee_color = (0, 0, 255)
                        wrong_count += 1
                    else:
                        self.knee_suggestion = "✅ Good knee angle!"
                        self.knee_color = (0, 255, 0)
                else:
                    self.knee_suggestion = "✅ Good standing knee angle!"
                    self.knee_color = (0, 255, 0)

                # Back suggestion (only if knees good)
                if 90 <= knee_angle <= 130:
                    right_angle_left = is_right_angle_at_knee(head, left_knee, right_knee)
                    right_angle_right = is_right_angle_at_knee(head, right_knee, left_knee)
                    if right_angle_left or right_angle_right:
                        self.back_suggestion = "🧍 Keep your back straight!"
                        self.back_color = (0, 0, 255)
                        wrong_count += 1
                    else:
                        self.back_suggestion = "✅ Back alignment good!"
                        self.back_color = (0, 255, 0)
                else:
                    self.back_suggestion = ""

                # Deduct score
                self.score = max(0, self.score - wrong_count)

            # Drawing
            cv2.putText(img, f"Reps: {self.reps}/{self.target_reps}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, f"Score: {self.score}", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if self.finished:
                cv2.putText(img, "🏁 Finished!", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                if self.knee_suggestion:
                    cv2.putText(img, self.knee_suggestion, (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.knee_color, 2)
                if self.back_suggestion:
                    cv2.putText(img, self.back_suggestion, (20, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.back_color, 2)

            knee_circle_color = (0, 255, 0) if 90 <= knee_angle <= 130 else (0, 0, 255)
            cv2.circle(img, left_knee, 8, knee_circle_color, -1)
            cv2.putText(img, f"{int(knee_angle)}°",
                       (left_knee[0] + 10, left_knee[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, knee_circle_color, 2)

        return VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------
# Main App
# ----------------------
def main_app():
    tabs = ["Workouts", "Stats", "Logout"]
    choice = st.sidebar.selectbox("Navigation", tabs)

    if choice == "Workouts":
        st.subheader("💪 Workouts")
        
        st.session_state.target_reps = st.number_input(
            "Enter number of reps",
            min_value=1,
            max_value=50,
            value=st.session_state.target_reps
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Squats"):
                st.session_state.start_squats = True
                # Clear the queue
                while not st.session_state.workout_result_queue.empty():
                    try:
                        st.session_state.workout_result_queue.get_nowait()
                    except queue.Empty:
                        break
                st.rerun()

        with col2:
            if st.button("Stop Workout"):
                st.session_state.start_squats = False
                st.rerun()

        if st.session_state.start_squats:
            # Capture target_reps value before creating the factory
            target_reps_value = st.session_state.target_reps
            result_queue = st.session_state.workout_result_queue
            
            webrtc_ctx = webrtc_streamer(
                key="squat-workout",
                video_processor_factory=lambda: PoseCoach(
                    target_reps=target_reps_value,
                    result_queue=result_queue
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }),
                mode=WebRtcMode.SENDRECV,
            )
            
            # Check for completed workout in queue
            # Check for completed workout in queue
            try:
                result = st.session_state.workout_result_queue.get_nowait()
                if result.get("finished"):
                    # Save the workout data
                    workout_record = {
                        "user": st.session_state.current_user,
                        "exercise": "squats", 
                        "reps": result["reps"],
                        "score": result["score"],
                        "timestamp": pd.Timestamp.now()
                    }
                    st.session_state.workout_data.append(workout_record)
                    
                    save_workout_data(st.session_state.workout_data)  # <-- ADD THIS LINE
                    
                    st.success(f"🎉 Workout completed! Reps: {result['reps']}, Score: {result['score']}")
                    st.balloons()
                    
                    if st.button("Start New Workout"):
                        st.session_state.start_squats = False
                        st.rerun()
            except queue.Empty:
                pass
            
        else:
            st.info("Click 'Start Squats' to begin your workout!")

    elif choice == "Stats":
        st.subheader("📊 Stats")
        data = st.session_state.workout_data
        
        if data:
            df = pd.DataFrame(data)
            user_df = df[df["user"] == st.session_state.current_user]
            
            if not user_df.empty:
                st.dataframe(user_df)
                
                chart = alt.Chart(user_df).mark_bar().encode(
                    x='timestamp:T',
                    y='reps:Q',
                    color='score:Q',
                    tooltip=['exercise', 'reps', 'score', 'timestamp']
                ).properties(title="Workout History")
                st.altair_chart(chart, use_container_width=True)
                
                st.subheader("Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Workouts", len(user_df))
                c2.metric("Total Reps", user_df['reps'].sum())
                c3.metric("Average Score", round(user_df['score'].mean(), 1))
            else:
                st.info("No workout data yet. Complete a workout to see stats here!")
        else:
            st.info("No workout data yet. Complete a workout to see stats here!")

    elif choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.start_squats = False
        st.success("Logged out successfully!")
        st.rerun()

# Run App
if not st.session_state.logged_in:
    login_screen()
else:
    main_app()