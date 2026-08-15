import json
import cv2 as cv
from omrchecker.engine import OMRReader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# 1. SETUP PATHS
TEMPLATE_PATH = "template.json"
FILLED_IMAGE_PATH = r'D:\opencv-practice\download - Copy.jfif'

# 2. MARKING SCHEME & ANSWER KEY (180 Questions)
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

# 3. RUN OPEN-SOURCE OMR READER
omr_reader = OMRReader(TEMPLATE_PATH)
results, debug_image = omr_reader.process_image(FILLED_IMAGE_PATH)

# Extract parsed data
roll_number = results.get("RollNumber", "UNKNOWN")
student_answers = results.get("QuestionsBlock1", {})

# 4. EVALUATE SCORES
correct_count = 0
wrong_count = 0
blank_count = 0
multiple_count = 0
total_score = 0

for q_num, correct_ans in ANSWER_KEY.items():
    ans = student_answers.get(q_num, "BLANK")
    if ans == "BLANK" or ans == "":
        blank_count += 1
        total_score += MARKS_BLANK
    elif ans == "MULTIPLE" or len(ans) > 1:
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

# 5. RICH DISPLAY & REPORT EXPORT
console = Console(record=True)

header_panel = Panel(
    "[bold cyan]MAYANK CODE EXAM[/bold cyan]\n[dim white]Open-Source OMR Evaluation System[/dim white]",
    style="bold blue",
    box=box.DOUBLE,
    expand=False,
    align="center"
)

info_table = Table(show_header=False, box=box.SIMPLE_HEAD, padding=(0, 2))
info_table.add_column("Key", style="bold yellow")
info_table.add_column("Value", style="bold white")
info_table.add_row("Roll Number", f"[bold green]{roll_number}[/bold green]")

score_table = Table(title="[bold magenta]Performance Summary[/bold magenta]", box=box.ROUNDED)
score_table.add_column("Category", style="cyan", justify="left")
score_table.add_column("Count", style="bold white", justify="center")
score_table.add_column("Impact", style="bold", justify="right")

score_table.add_row("Total Questions", str(total_questions), "[white]100%[/white]")
score_table.add_row("Correct (+4)", str(correct_count), f"[green]+{correct_count * MARKS_CORRECT}[/green]")
score_table.add_row("Wrong (-1)", str(wrong_count), f"[red]{wrong_count * MARKS_WRONG}[/red]")
score_table.add_row("Blank (0)", str(blank_count), "[gray]0[/gray]")
score_table.add_row("Multiple Marked (0)", str(multiple_count), "[yellow]0[/yellow]")

percentage_color = "green" if percentage >= 60 else ("yellow" if percentage >= 40 else "red")

result_panel = Panel(
    f"[bold white]Final Score:[/bold white] [bold cyan]{total_score}[/bold cyan] / [bold white]{max_possible_marks}[/bold white]\n"
    f"[bold white]Percentage :[/bold white] [{percentage_color}]{percentage:.2f}%[/{percentage_color}]",
    title="[bold gold1]Result[/bold gold1]",
    border_style=percentage_color,
    box=box.ROUNDED,
    expand=False
)

console.print()
console.print(header_panel)
console.print(info_table)
console.print(score_table)
console.print(result_panel)
console.print()

# Export report to HTML
console.save_html("omr_report.html")

# Show debug overlay generated by the framework
cv.imshow("Processed OMR Sheet", debug_image)
cv.waitKey(0)
cv.destroyAllWindows()
