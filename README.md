# Person Tracking with Flow Line Visualization

## 🎯 Objective

The objective of this project is to **track a specific person** in a provided video and **generate a flow line** that visualizes their movement throughout the video.

### 📌 Task Details

- **Input Video**: Provided in the project.
- **Target Person**: The individual holding a **blue trolley with blue plants**, wearing a **green jacket** and a **white shirt**.
- **Reference Image**: Used to assist in identifying and isolating the target person.
- **Expected Output**:
  - A **video** with the target person tracked frame-by-frame.
  - A **flow line** overlaid to represent their movement path.
  - The **timestamp** when the person first appears in the video.
  - The **codebase** used to produce this output.

---

## 🧠 System Design

PDF -: https://drive.google.com/file/d/1Iv4jkDk6rLzc1W18S7zIJQ3cu7z9YogP/view?usp=sharing

### Step-by-step Pipeline

1. **Input Collection**:
   - Read the video frame by frame using **OpenCV**.
   - Load the reference image of the target person.

2. **Target Person Detection**:
   - Use **YOLOv8** (Ultralytics) for person detection.
   - Match detected persons with the reference image using visual similarity techniques (e.g., CLIP or cosine similarity).
   - Identify and extract the target person based on both **text** and **image** cues.

3. **Real-time Tracking**:
   - Use **DeepSORT** to assign consistent IDs to tracked people across frames.
   - Focus on tracking the **target person ID**.

4. **Flow Line Generation**:
   - Maintain a list of tracked coordinates (centroids).
   - Draw a **trajectory line** across frames to show movement.

5. **Timestamp Detection**:
   - Detect and print the **timestamp** (in seconds or HH:MM:SS) when the target person **first appears** in the video.

6. **Output**:
   - Annotated video with tracking boxes, ID, and movement path.
   - Logs and timestamp of first appearance.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- PyTorch
- YOLOv8 (Ultralytics)
- DeepSORT
- CLIP (optional for reference image-text matching)

---

## 🚀 Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/person-tracking-flowline.git
cd person-tracking-flowline
```
### Step 1: Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```
### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```
### Step 4: Run the Script

```bash
python main.py
```

## Output
 Video -: https://drive.google.com/file/d/1arpYI_atR48H2h13gT0VrH61MFJVYjI3/view?usp=sharing


