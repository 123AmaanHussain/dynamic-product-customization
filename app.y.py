import os
import cv2
from fer import FER
from transformers import pipeline
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
app.secret_key = 'your_secret_key'
socketio = SocketIO(app)

# Initialize the emotion detector and sentiment analysis model
detector = FER(mtcnn=True)
text_emotion_model = pipeline('sentiment-analysis')

# Sample product database categorized by mood
products_by_mood = {
    "happy": [
        {"id": 1, "name": "Product A", "category": "Electronics"},
        {"id": 2, "name": "Product B", "category": "Books"},
    ],
    "sad": [
        {"id": 3, "name": "Product C", "category": "Clothing"},
        {"id": 4, "name": "Product D", "category": "Books"},
    ],
    "angry": [
        {"id": 5, "name": "Product E", "category": "Electronics"},
        {"id": 6, "name": "Product F", "category": "Clothing"},
    ],
    "neutral": [
        {"id": 7, "name": "Product G", "category": "Accessories"},
        {"id": 8, "name": "Product H", "category": "Home Goods"},
    ]
}

# In-memory user data store
users = {
    "testuser": "password",
    "friend1": "password",
    "friend2": "password",
    "friend3": "password"
}
user_profiles = {
    "testuser": {"friends": ["friend1"], "chats": []},
    "friend1": {"friends": ["testuser"], "chats": []},
    "friend2": {"friends": [], "chats": []},
    "friend3": {"friends": [], "chats": []}
}
activity_feed = []
selected_products = []

# Global variables to control the camera thread
camera_thread = None
camera_running = False

# Function to capture video frames and detect emotions
def gen_frames():
    global camera_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotions in the frame
        emotions = detector.detect_emotions(frame)
        dominant_emotion = get_dominant_emotion(emotions)
        
        # Generate recommendations based on detected emotion
        recommendations, chatbot_response = adapt_interface_and_interactions(dominant_emotion, selected_products)
        socketio.emit('update_recommendations', {'recommendations': recommendations, 'chatbot_response': chatbot_response})

        # Draw bounding boxes and emotions on the frame
        for emotion in emotions:
            box = emotion['box']
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            emotion_text = max(emotion['emotions'], key=emotion['emotions'].get)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def get_dominant_emotion(emotions):
    if not emotions:
        return "neutral"
    
    emotion_scores = emotions[0]['emotions']
    
    # Return the most dominant emotion
    return max(emotion_scores, key=emotion_scores.get)

# Flask route to provide the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to start the camera
@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_running, camera_thread
    if not camera_running:
        camera_running = True
        camera_thread = threading.Thread(target=gen_frames)
        camera_thread.start()
    return jsonify(success=True)

# Flask route to stop the camera
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running
    camera_running = False
    return jsonify(success=True)

# Flask route to provide the cart items
@app.route('/get_cart_items', methods=['POST'])
def get_cart_items():
    data = request.json
    selected_products = data.get('selected_products', [])
    cart_items = [p for p in [p for mood in products_by_mood.values() for p in mood] if p['id'] in selected_products]
    return jsonify(cart_items=cart_items)

def adapt_interface_and_interactions(dominant_emotion, selected_products):
    recommendations = products_by_mood.get(dominant_emotion, [])
    if dominant_emotion == "happy":
        chatbot_response = "It seems like you're in a great mood! Check out these exciting products!"
    elif dominant_emotion == "sad":
        chatbot_response = "I'm here to help. Maybe these products can improve your mood."
    elif dominant_emotion == "angry":
        chatbot_response = "I understand your frustration. Here are some items that might help."
    else:
        chatbot_response = "Here are some products you might like."

    return recommendations, chatbot_response

# User authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return "Invalid credentials"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Add friend route
@app.route('/add_friend', methods=['POST'])
def add_friend():
    if 'username' not in session:
        return jsonify(success=False, error="User not logged in")
    username = session['username']
    friend = request.json.get('friend')
    if friend in users:
        if friend not in user_profiles[username]['friends']:
            user_profiles[username]['friends'].append(friend)
            user_profiles[friend]['friends'].append(username)
            return jsonify(success=True)
        else:
            return jsonify(success=False, error="Already friends")
    else:
        return jsonify(success=False, error="Friend not found")

# Share product route
@app.route('/share_product', methods=['POST'])
def share_product():
    if 'username' not in session:
        return jsonify(success=False, error="User not logged in")
    username = session['username']
    product_id = request.json['product_id']
    product = next((p for p in [p for mood in products_by_mood.values() for p in mood] if p['id'] == product_id), None)
    if product:
        activity_feed.append({"user": username, "product": product})
        return jsonify(success=True)
    else:
        return jsonify(success=False, error="Product not found")

# Social feed route
@app.route('/social_feed')
def social_feed():
    return jsonify(activity_feed)

# Chat route
@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return jsonify(success=False, error="User not logged in")
    username = session['username']
    message = request.json['message']
    recipient = request.json['recipient']
    if recipient in users:
        user_profiles[recipient]['chats'].append({"from": username, "message": message})
        return jsonify(success=True)
    else:
        return jsonify(success=False, error="Recipient not found")

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
