import streamlit as st
import cv2 as cv
import numpy as np
import os
import pandas as pd
from datetime import datetime
import base64
from pathlib import Path

# CONSTANTS & ANSWER KEY 
FILL_THRESHOLD = 0.35
MARKS_CORRECT = 4
MARKS_WRONG = -1
MARKS_BLANK = 0
MARKS_MULTIPLE = 0

ANSWER_KEY = {
    1: "C", 2: "A", 3: "D", 4: "B", 5: "A", 6: "D", 7: "C", 8: "B", 9: "A", 10: "D",
    11: "B", 12: "C", 13: "A", 14: "D", 15: "B", 16: "C", 17: "A", 18: "D", 19: "B", 20: "C",
    21: "D", 22: "A", 23: "B", 24: "C", 25: "A", 26: "D", 27: "B", 28: "C", 29: "D", 30: "A",
    31: "B", 32: "D", 33: "C", 34: "A", 35: "B", 36: "C", 37: "D", 38: "A", 39: "C", 40: "B",
    41: "A", 42: "D", 43: "B", 44: "C", 45: "A", 46: "B", 47: "D", 48: "C", 49: "A", 50: "D",
    51: "C", 52: "B", 53: "A", 54: "D", 55: "C", 56: "A", 57: "B", 58: "D", 59: "C", 60: "A",
    61: "D", 62: "B", 63: "C", 64: "A", 65: "D", 66: "B", 67: "C", 68: "A", 69: "D", 70: "B",
    71: "C", 72: "A", 73: "D", 74: "B", 75: "C", 76: "A", 77: "D", 78: "B", 79: "C", 80: "A",
    81: "D", 82: "B", 83: "C", 84: "A", 85: "D", 86: "C", 87: "B", 88: "A", 89: "D", 90: "C",
    91: "B", 92: "A", 93: "D", 94: "C", 95: "B", 96: "A", 97: "D", 98: "C", 99: "B", 100: "A",
    101: "D", 102: "C", 103: "A", 104: "B", 105: "D", 106: "C", 107: "B", 108: "A", 109: "D", 110: "C",
    111: "B", 112: "A", 113: "D", 114: "C", 115: "B", 116: "A", 117: "D", 118: "C", 119: "B", 120: "A",
    121: "C", 122: "D", 123: "B", 124: "A", 125: "C", 126: "D", 127: "A", 128: "B", 129: "C", 130: "D",
    131: "A", 132: "B", 133: "D", 134: "C", 135: "A", 136: "B", 137: "C", 138: "D", 139: "A", 140: "B",
    141: "C", 142: "D", 143: "A", 144: "B", 145: "C", 146: "D", 147: "A", 148: "B", 149: "C", 150: "D",
    151: "B", 152: "A", 153: "C", 154: "D", 155: "B", 156: "A", 157: "C", 158: "D", 159: "A", 160: "B",
    161: "C", 162: "D", 163: "A", 164: "B", 165: "C", 166: "D", 167: "A", 168: "B", 169: "C", 170: "D",
    171: "A", 172: "B", 173: "C", 174: "D", 175: "A", 176: "B", 177: "C", 178: "D", 179: "A", 180: "B"
}

# ----------------------------------------------------------------------
# HELPER FUNCTIONS (unchanged)
# ----------------------------------------------------------------------
def get_fill_ratio(image, x, y, radius=5):
    roi = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    if roi.size == 0:
        return 0.0
    mask = np.zeros(roi.shape, dtype=np.uint8)
    center = (radius, radius)
    cv.circle(mask, center, radius, 255, -1)
    pixels = roi[mask == 255]
    if len(pixels) == 0:
        return 0.0
    dark_pixels = np.sum(pixels < 150)
    return dark_pixels / len(pixels)

def extract_grid_bubbles(area_gray, min_w=10, max_w=20, min_h=10, max_h=20):
    _, thres = cv.threshold(area_gray, 200, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bubble_centers = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if (min_w <= w <= max_w and min_h <= h <= max_h and
            0.85 <= w / h <= 1.15 and 50 <= area <= 180 and 0.4 <= circularity <= 1.3):
            bubble_centers.append((x + w // 2, y + h // 2))
    return bubble_centers

def organize_into_rows(bubbles, y_threshold=5):
    bubbles = sorted(bubbles, key=lambda p: p[1])
    rows = []
    current_row = []
    previous_y = None
    for x, y in bubbles:
        if previous_y is None or abs(y - previous_y) <= y_threshold:
            current_row.append((x, y))
        else:
            rows.append(current_row)
            current_row = [(x, y)]
        previous_y = y
    if current_row:
        rows.append(current_row)
    for row in rows:
        row.sort(key=lambda p: p[0])
    return rows

def parse_digit_grid(blank_crop, filled_crop):
    centers = extract_grid_bubbles(blank_crop)
    rows = organize_into_rows(centers)
    if not rows:
        return "UNKNOWN"
    num_cols = len(rows[0])
    digits = []
    for col_idx in range(num_cols):
        scores = []
        for row_idx, row in enumerate(rows):
            if col_idx < len(row):
                x, y = row[col_idx]
                ratio = get_fill_ratio(filled_crop, x, y)
                scores.append((row_idx, ratio))
        filled_digits = [digit for digit, ratio in scores if ratio >= FILL_THRESHOLD]
        if len(filled_digits) == 1:
            digits.append(str(filled_digits[0]))
        elif len(filled_digits) > 1:
            digits.append("X")
        else:
            digits.append("?")
    return "".join(digits)

def get_question_bubbles(rows):
    questions = {}
    for row_index, row in enumerate(rows):
        for block in range(6):
            start = block * 4
            end = start + 4
            question_number = block * 30 + row_index + 1
            questions[question_number] = row[start:end]
    return questions

# ----------------------------------------------------------------------
# HTML REPORT GENERATOR
# ----------------------------------------------------------------------
def generate_html_report(roll_number, test_id, student_answers, correct_count, wrong_count, 
                         blank_count, multiple_count, total_score, max_possible_marks, 
                         percentage, question_details):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OMR Evaluation Report - {roll_number}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; 
                         border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; border-bottom: 3px solid #4CAF50; padding-bottom: 20px; }}
            .header h1 {{ color: #333; margin: 0; }}
            .header h2 {{ color: #666; font-weight: normal; }}
            .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .info-item {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
            .info-item label {{ font-weight: bold; color: #555; }}
            .score-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
            .score-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
            .score-card.green {{ background: #d4edda; border: 2px solid #28a745; }}
            .score-card.yellow {{ background: #fff3cd; border: 2px solid #ffc107; }}
            .score-card.red {{ background: #f8d7da; border: 2px solid #dc3545; }}
            .score-card .number {{ font-size: 32px; font-weight: bold; }}
            .score-card .label {{ color: #666; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background: #f5f5f5; }}
            .correct {{ color: green; font-weight: bold; }}
            .wrong {{ color: red; font-weight: bold; }}
            .blank {{ color: gray; }}
            .multiple {{ color: orange; font-weight: bold; }}
            .footer {{ text-align: center; margin-top: 30px; color: #999; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📝 OMR Evaluation Report</h1>
                <h2>Automated Answer Sheet Evaluation</h2>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="info-grid">
                <div class="info-item">
                    <label>Roll Number:</label><br>
                    <span style="font-size: 24px; font-weight: bold;">{roll_number}</span>
                </div>
                <div class="info-item">
                    <label>Test ID:</label><br>
                    <span style="font-size: 24px; font-weight: bold;">{test_id}</span>
                </div>
            </div>
            
            <div class="score-grid">
                <div class="score-card green">
                    <div class="number">{total_score}</div>
                    <div class="label">Total Score (out of {max_possible_marks})</div>
                </div>
                <div class="score-card yellow">
                    <div class="number">{percentage:.1f}%</div>
                    <div class="label">Percentage</div>
                </div>
                <div class="score-card green">
                    <div class="number">{correct_count}</div>
                    <div class="label">Correct Answers</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0;">
                <div style="background: #d4edda; padding: 10px; text-align: center; border-radius: 5px;">
                    <div><strong>✅ Correct</strong></div>
                    <div style="font-size: 24px;">{correct_count}</div>
                </div>
                <div style="background: #f8d7da; padding: 10px; text-align: center; border-radius: 5px;">
                    <div><strong>❌ Wrong</strong></div>
                    <div style="font-size: 24px;">{wrong_count}</div>
                </div>
                <div style="background: #e2e3e5; padding: 10px; text-align: center; border-radius: 5px;">
                    <div><strong>⬜ Blank</strong></div>
                    <div style="font-size: 24px;">{blank_count}</div>
                </div>
                <div style="background: #fff3cd; padding: 10px; text-align: center; border-radius: 5px;">
                    <div><strong>⚠️ Multiple</strong></div>
                    <div style="font-size: 24px;">{multiple_count}</div>
                </div>
            </div>
            
            <h3>📋 Question-wise Analysis</h3>
            <table>
                <thead>
                    <tr>
                        <th>Q.No</th>
                        <th>Student Answer</th>
                        <th>Correct Answer</th>
                        <th>Status</th>
                        <th>Marks</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for q_num, details in question_details.items():
        status_class = details['status'].lower()
        status_text = details['status']
        if status_text == "Correct":
            status_text = "✅ " + status_text
        elif status_text == "Wrong":
            status_text = "❌ " + status_text
        elif status_text == "Blank":
            status_text = "⬜ " + status_text
        else:
            status_text = "⚠️ " + status_text
            
        html += f"""
                    <tr>
                        <td>{q_num}</td>
                        <td>{details['student_answer']}</td>
                        <td>{details['correct_answer']}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{details['marks']}</td>
                    </tr>
        """
    
    html += """
                </tbody>
            </table>
            
            <div class="footer">
                <p>This is a computer-generated report. Please verify with the original OMR sheet.</p>
                <p>Generated by Automated OMR Evaluation System</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

# DOWNLOAD FUNCTIONS
def get_download_link(data, filename, mime_type):
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'

def generate_marksheet_csv(roll_number, test_id, question_details, total_score, max_possible_marks, percentage):
    data = []
    for q_num, details in question_details.items():
        data.append({
            'Question': q_num,
            'Student Answer': details['student_answer'],
            'Correct Answer': details['correct_answer'],
            'Status': details['status'],
            'Marks': details['marks']
        })
    
    # Add summary rows
    data.append({})
    data.append({'Question': 'SUMMARY', 'Student Answer': '', 'Correct Answer': '', 'Status': '', 'Marks': ''})
    data.append({'Question': 'Roll Number', 'Student Answer': roll_number, 'Correct Answer': '', 'Status': '', 'Marks': ''})
    data.append({'Question': 'Test ID', 'Student Answer': test_id, 'Correct Answer': '', 'Status': '', 'Marks': ''})
    data.append({'Question': 'Total Score', 'Student Answer': f'{total_score}/{max_possible_marks}', 'Correct Answer': '', 'Status': '', 'Marks': ''})
    data.append({'Question': 'Percentage', 'Student Answer': f'{percentage:.2f}%', 'Correct Answer': '', 'Status': '', 'Marks': ''})
    
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    return csv.encode('utf-8')

def display_full_omr_with_annotations(original_image, questions, student_answers, answer_key):
    """Display full OMR sheet with annotations"""
    img_display = original_image.copy()
    
    # Define the answer area coordinates (same as in processing)
    x_start, x_end = 70, 690
    y_start, y_end = 400, 980
    
    # Draw a rectangle around the answer area
    cv.rectangle(img_display, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv.putText(img_display, "Answer Area", (x_start, y_start-10), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    options_map = ["A", "B", "C", "D"]
    
    # Annotate the full image with the evaluation results
    for q_num, q_bubbles in questions.items():
        student_ans = student_answers[q_num]
        correct_ans = answer_key.get(q_num)
        
        for opt_idx, (x, y) in enumerate(q_bubbles):
            # Adjust coordinates to match the full image
            full_x = x + x_start
            full_y = y + y_start
            
            option_char = options_map[opt_idx]
            if option_char == student_ans:
                color = (0, 255, 0) if student_ans == correct_ans else (0, 0, 255)
                cv.circle(img_display, (full_x, full_y), 6, color, -1)
            else:
                cv.circle(img_display, (full_x, full_y), 3, (200, 200, 200), 1)
    
    # Add info overlay
    info_text = f"Roll: {roll_number} | Test: {test_id} | Score: {total_score}/{max_possible_marks}"
    cv.putText(img_display, info_text, (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return img_display

# STREAMLIT APP
st.set_page_config(page_title="OMR Evaluation System", layout="wide")
st.title("📝 Automated OMR Evaluation System")
st.markdown("Upload the **filled** OMR sheet. The blank template is static and loaded automatically.")

# ---- Load the static blank image ----
current_dir = Path(__file__).parent
BLANK_IMAGE_PATH = current_dir / "download.jfif"

if not os.path.exists(BLANK_IMAGE_PATH):
    st.error(f"Blank image not found at '{BLANK_IMAGE_PATH}'. Please place the blank OMR image in the app directory.")
    st.stop()

blank_img = cv.imread(BLANK_IMAGE_PATH)
if blank_img is None:
    st.error("Failed to load blank image. Ensure it is a valid image file.")
    st.stop()

blank_gray = cv.cvtColor(blank_img, cv.COLOR_BGR2GRAY)

# ---- Upload filled image ----
filled_file = st.file_uploader("Upload **Filled** OMR", type=["png", "jpg", "jpeg",'jfif'])

if filled_file is not None:
    # Read filled image
    filled_bytes = np.asarray(bytearray(filled_file.read()), dtype=np.uint8)
    filled_img = cv.imdecode(filled_bytes, cv.IMREAD_COLOR)
    if filled_img is None:
        st.error("Failed to decode filled image. Please upload a valid image file.")
        st.stop()

    filled_gray = cv.cvtColor(filled_img, cv.COLOR_BGR2GRAY)

    # 1. Roll Number & Test ID
    roll_area_blank = blank_gray[155:380, 35:240]
    roll_area_filled = filled_gray[155:380, 35:240]
    test_id_area_blank = blank_gray[155:380, 250:360]
    test_id_area_filled = filled_gray[155:380, 250:360]

    roll_number = parse_digit_grid(roll_area_blank, roll_area_filled)
    test_id = parse_digit_grid(test_id_area_blank, test_id_area_filled)

    # 2. Answer Area
    blank_answer_area = blank_gray[400:980, 70:690]
    filled_answer_area = filled_gray[400:980, 70:690]

    bubble_centers = extract_grid_bubbles(blank_answer_area)
    rows = organize_into_rows(bubble_centers)
    questions = get_question_bubbles(rows)

    options_map = ["A", "B", "C", "D"]
    student_answers = {}

    for q_num, q_bubbles in questions.items():
        marked_options = []
        for opt_idx, (x, y) in enumerate(q_bubbles):
            ratio = get_fill_ratio(filled_answer_area, x, y)
            if ratio >= FILL_THRESHOLD:
                marked_options.append(options_map[opt_idx])
        if len(marked_options) == 1:
            student_answers[q_num] = marked_options[0]
        elif len(marked_options) > 1:
            student_answers[q_num] = "MULTIPLE"
        else:
            student_answers[q_num] = "BLANK"

    # 3. Score Calculation
    correct_count = wrong_count = blank_count = multiple_count = 0
    total_score = 0
    question_details = {}

    for q_num, correct_ans in ANSWER_KEY.items():
        ans = student_answers.get(q_num, "BLANK")
        marks = 0
        status = "Unknown"
        
        if ans == "BLANK":
            blank_count += 1
            marks = MARKS_BLANK
            status = "Blank"
        elif ans == "MULTIPLE":
            multiple_count += 1
            marks = MARKS_MULTIPLE
            status = "Multiple"
        elif ans == correct_ans:
            correct_count += 1
            marks = MARKS_CORRECT
            status = "Correct"
        else:
            wrong_count += 1
            marks = MARKS_WRONG
            status = "Wrong"
        
        total_score += marks
        question_details[q_num] = {
            'student_answer': ans,
            'correct_answer': correct_ans,
            'status': status,
            'marks': marks
        }

    total_questions = len(ANSWER_KEY)
    max_possible_marks = total_questions * MARKS_CORRECT
    percentage = (total_score / max_possible_marks) * 100 if max_possible_marks > 0 else 0

    # 4. Display Full OMR with Annotations
    st.subheader("📄 Full OMR Sheet with Annotations")
    
    annotated_full = display_full_omr_with_annotations(filled_img, questions, student_answers, ANSWER_KEY)
    st.image(annotated_full, channels="BGR", use_container_width=True)
    
    st.info("✅ Green circles = Correct answers | ❌ Red circles = Wrong answers | ⚪ Gray circles = Not selected")

    # 5. Display Results
    st.divider()
    col_info, col_score = st.columns(2)

    with col_info:
        st.subheader("📋 Candidate Information")
        st.metric("Roll Number", roll_number)
        st.metric("Test ID", test_id)

    with col_score:
        st.subheader("📊 Score Summary")
        st.metric("Total Score", f"{total_score} / {max_possible_marks}")
        st.metric("Percentage", f"{percentage:.2f}%", delta=None)

    # Breakdown table
    st.subheader("🔍 Performance Breakdown")
    breakdown_data = {
        "Category": ["Total Questions", "Correct (+4)", "Wrong (-1)", "Blank (0)", "Multiple (0)"],
        "Count": [total_questions, correct_count, wrong_count, blank_count, multiple_count],
        "Impact": [
            f"{total_questions}",
            f"+{correct_count * MARKS_CORRECT}",
            f"{wrong_count * MARKS_WRONG}",
            "0",
            "0"
        ]
    }
    st.table(breakdown_data)

    # Result panel 
    if percentage >= 60:
        border_color = "green"
        emoji = "✅"
    elif percentage >= 40:
        border_color = "orange"
        emoji = "⚠️"
    else:
        border_color = "red"
        emoji = "❌"

    st.markdown(
        f"""
        <div style="border: 3px solid {border_color}; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f9f9f9;">
            <h2 style="text-align: center; color: {border_color};">{emoji} Final Result</h2>
            <p style="text-align: center; font-size: 24px; font-weight: bold; color: {border_color};">
                {total_score} / {max_possible_marks} &nbsp;|&nbsp; {percentage:.2f}%
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 6. Download Options
    st.divider()
    st.subheader("📥 Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Generate HTML report
        html_report = generate_html_report(
            roll_number, test_id, student_answers, correct_count, wrong_count,
            blank_count, multiple_count, total_score, max_possible_marks,
            percentage, question_details
        )
        html_bytes = html_report.encode('utf-8')
        
        st.download_button(
            label="📄 Download HTML Report",
            data=html_bytes,
            file_name=f"OMR_Report_{roll_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )
    
    with col2:
        # Generate CSV marksheet
        csv_data = generate_marksheet_csv(
            roll_number, test_id, question_details, total_score, max_possible_marks, percentage
        )
        
        st.download_button(
            label="📊 Download CSV Marksheet",
            data=csv_data,
            file_name=f"OMR_Marksheet_{roll_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Download annotated image
        is_success, img_buffer = cv.imencode(".png", annotated_full)
        if is_success:
            img_bytes = img_buffer.tobytes()
            st.download_button(
                label="🖼️ Download Annotated Image",
                data=img_bytes,
                file_name=f"OMR_Annotated_{roll_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )

    # 7. Detailed Question-wise Analysis
    with st.expander("📋 Show Detailed Question-wise Analysis"):
    # Create a dataframe for better display
        df_data = []
        for q_num, details in question_details.items():
            df_data.append({
                'Question': q_num,
                'Your Answer': details['student_answer'],
                'Correct Answer': details['correct_answer'],
                'Status': details['status'],
                'Marks': details['marks']
            })
        
        df = pd.DataFrame(df_data)
        
        # Color code the status using map (new method)
        def color_status(val):
            if val == 'Correct':
                return 'background-color: #d4edda'
            elif val == 'Wrong':
                return 'background-color: #f8d7da'
            elif val == 'Blank':
                return 'background-color: #e2e3e5'
            else:
                return 'background-color: #fff3cd'
        
        # Use map instead of applymap
        styled_df = df.style.map(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)

else:
    st.info("👆 Please upload the filled OMR image to begin evaluation.")