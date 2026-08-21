import cv2 as cv
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console(record=True)

#HEADER PANEL
header_panel = Panel(
    "[bold cyan]MAYANK CODE EXAM[/bold cyan]\n[dim white]Automated OMR Evaluation System[/dim white]",
    style="bold blue",
    box=box.DOUBLE,
    expand=False,
)

# 1. FILE PATHS & SETUP
BLANK_IMAGE_PATH = r'D:\opencv-practice\Blank_file.jfif'
FILLED_IMAGE_PATH = r'D:\opencv-practice\Dummy_answers.jfif'

blank_img = cv.imread(BLANK_IMAGE_PATH)
filled_img = cv.imread(FILLED_IMAGE_PATH)

blank_gray = cv.cvtColor(blank_img, cv.COLOR_BGR2GRAY)
filled_gray = cv.cvtColor(filled_img, cv.COLOR_BGR2GRAY)

FILL_THRESHOLD = 0.35

# 2. MARKING SCHEME & ANSWER KEY
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

MARKS_CORRECT = 4
MARKS_WRONG = -1
MARKS_BLANK = 0
MARKS_MULTIPLE = 0

# 3. HELPER FUNCTIONS

def get_fill_ratio(image, x, y, radius=5):
    """Calculates the fill ratio of dark pixels inside a circular ROI."""
    roi = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    if roi.size == 0:
        return 0.0

    mask = np.zeros(roi.shape, dtype=np.uint8)
    center = (radius, radius)
    cv.circle(mask, center, radius, 255, -1)

    pixels = roi[mask == 255]
    if len(pixels) == 0:
        return 0.0

    # Count dark pixels 
    dark_pixels = np.sum(pixels < 150)
    return dark_pixels / len(pixels)


def extract_grid_bubbles(area_gray, min_w=10, max_w=20, min_h=10, max_h=20):
    """Extracts valid bubble center coordinates from a given cropped region."""
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

        if (
            min_w <= w <= max_w
            and min_h <= h <= max_h
            and 0.85 <= w / h <= 1.15
            and 50 <= area <= 180
            and 0.4 <= circularity <= 1.3
        ):
            center_x = x + w // 2
            center_y = y + h // 2
            bubble_centers.append((center_x, center_y))

    return bubble_centers


def organize_into_rows(bubbles, y_threshold=5):
    """Sorts bubbles into rows based on Y coordinates."""
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

    # Sort each row left to right (by X)
    for row in rows:
        row.sort(key=lambda p: p[0])

    return rows


# 4. ROLL NUMBER & TEST ID EXTRACTION

roll_area_blank = blank_gray[155:380, 35:240]
roll_area_filled = filled_gray[155:380, 35:240]

test_id_area_blank = blank_gray[155:380, 250:360]
test_id_area_filled = filled_gray[155:380, 250:360]

def parse_digit_grid(blank_crop, filled_crop):
    """Parses column-based digit grids (Roll No. / Test ID)."""
    centers = extract_grid_bubbles(blank_crop)
    rows = organize_into_rows(centers)

    if not rows:
        return "UNKNOWN"

    # Transpose rows into columns
    num_cols = len(rows[0])
    digits = []

    for col_idx in range(num_cols):
        scores = []
        for row_idx, row in enumerate(rows):
            if col_idx < len(row):
                x, y = row[col_idx]
                ratio = get_fill_ratio(filled_crop, x, y)
                scores.append((row_idx, ratio))

        # Find filled digits in this column
        filled_digits = [digit for digit, ratio in scores if ratio >= FILL_THRESHOLD]

        if len(filled_digits) == 1:
            digits.append(str(filled_digits[0]))
        elif len(filled_digits) > 1:
            digits.append("X")  # Multiple options marked
        else:
            digits.append("?")  # Blank / Unmarked

    return "".join(digits)


roll_number = parse_digit_grid(roll_area_blank, roll_area_filled)
test_id = parse_digit_grid(test_id_area_blank, test_id_area_filled)

# 5. ANSWER AREA PROCESSING (180 QUESTIONS)
blank_answer_area = blank_gray[400:980, 70:690]
filled_answer_area = filled_gray[400:980, 70:690]

bubble_centers = extract_grid_bubbles(blank_answer_area)
rows = organize_into_rows(bubble_centers)

def get_question_bubbles(rows):
    questions = {}
    for row_index, row in enumerate(rows):
        for block in range(6):
            start = block * 4
            end = start + 4
            question_number = block * 30 + row_index + 1
            questions[question_number] = row[start:end]
    return questions

questions = get_question_bubbles(rows)

# 6. EXTRACT STUDENT ANSWERS & EVALUATE
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

# Calculate Score
correct_count = 0
wrong_count = 0
blank_count = 0
multiple_count = 0
total_score = 0

for q_num, correct_ans in ANSWER_KEY.items():
    ans = student_answers.get(q_num, "BLANK")
    if ans == "BLANK":
        blank_count += 1
        total_score += MARKS_BLANK
    elif ans == "MULTIPLE":
        multiple_count += 1
        total_score += MARKS_MULTIPLE
    elif ans == correct_ans:
        correct_count += 1
        total_score += MARKS_CORRECT
    else:
        wrong_count += 1
        total_score += MARKS_WRONG

total_questions = len(ANSWER_KEY)
max_possible_marks = total_questions * MARKS_CORRECT
percentage = (total_score / max_possible_marks) * 100 if max_possible_marks > 0 else 0

# Visual debug window showing detected answers
debug_display = cv.cvtColor(filled_answer_area, cv.COLOR_GRAY2BGR)
for q_num, q_bubbles in questions.items():
    student_ans = student_answers[q_num]
    correct_ans = ANSWER_KEY.get(q_num)

    for opt_idx, (x, y) in enumerate(q_bubbles):
        option_char = options_map[opt_idx]
        if option_char == student_ans:
            # Green circle if correct, Red circle if wrong
            color = (0, 255, 0) if student_ans == correct_ans else (0, 0, 255)
            cv.circle(debug_display, (x, y), 6, color, -1)
        else:
            cv.circle(debug_display, (x, y), 3, (200, 200, 200), 1)

info_table = Table(show_header=False, box=box.SIMPLE_HEAD, padding=(0, 2))
info_table.add_column("Key", style="bold yellow")
info_table.add_column("Value", style="bold white")

info_table.add_row("Roll Number", f"[bold green]{roll_number}[/bold green]")
info_table.add_row("Test ID", f"[bold green]{test_id}[/bold green]")

# ---------------------------------------------------------
# 3. SCORE BREAKDOWN TABLE
# ---------------------------------------------------------
score_table = Table(title="[bold magenta]Performance Summary[/bold magenta]", box=box.ROUNDED)
score_table.add_column("Category", style="cyan", justify="left")
score_table.add_column("Count", style="bold white", justify="center")
score_table.add_column("Impact", style="bold", justify="right")

score_table.add_row("Total Questions", str(total_questions), "[white]100%[/white]")
score_table.add_row("Correct (+4)", str(correct_count), f"[green]+{correct_count * MARKS_CORRECT}[/green]")
score_table.add_row("Wrong (-1)", str(wrong_count), f"[red]{wrong_count * MARKS_WRONG}[/red]")
score_table.add_row("Blank (0)", str(blank_count), "[gray]0[/gray]")
score_table.add_row("Multiple Marked (0)", str(multiple_count), "[yellow]0[/yellow]")

# Color logic for final score/percentage
percentage_color = "green" if percentage >= 60 else ("yellow" if percentage >= 40 else "red")

# ---------------------------------------------------------
# 4. FINAL RESULT PANEL
# ---------------------------------------------------------
result_text = (
    f"[bold white]Final Score:[/bold white] [bold cyan]{total_score}[/bold cyan] / [bold white]{max_possible_marks}[/bold white]\n"
    f"[bold white]Percentage :[/bold white] [{percentage_color}]{percentage:.2f}%[/{percentage_color}]"
)

result_panel = Panel(
    result_text,
    title="[bold gold1]Result[/bold gold1]",
    border_style=percentage_color,
    box=box.ROUNDED,
    expand=False
)

# ---------------------------------------------------------
# RENDER ALL COMPONENTS TO TERMINAL
# ---------------------------------------------------------
console.print()
console.print(header_panel)
console.print(info_table)
console.print(score_table)
console.print(result_panel)
console.print()
console.save_html("terminal_report.html")

cv.imshow("Evaluated OMR Area", debug_display)
cv.waitKey(0)
cv.destroyAllWindows()