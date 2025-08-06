# app.py
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from main import run_tracking_pipeline
import os

app = Flask(__name__, static_folder="static")
CORS(app)

@app.route("/")
def serve_index():
    return send_file("index.html")  # serves index.html on root

@app.route("/run", methods=["GET"])
def run_processing():
    output_path = "static/output.avi"
    
    if os.path.exists(output_path):
        print("✅ Output already exists. Skipping reprocessing.")
        return jsonify({"status": "success", "output": "/static/output.avi"})
    
    success = run_tracking_pipeline(
        cropped_ref_img_path="static/Traget Person.png",
        video_path="static/Task_Video.mp4",
        output_path=output_path
    )

    if not success:
        return jsonify({"status": "error"}), 500

    return jsonify({"status": "success", "output": "/static/output.avi"})

@app.route("/static/output.avi")
def get_video():
    return send_file("static/output.avi", mimetype="video/x-msvideo")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
