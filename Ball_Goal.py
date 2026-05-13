from collections import deque
import numpy as np
import cv2

# ── Config ──────────────────────────────────────────────
STREAM_URL          = "http://192.168.1.8:8080/video"
TARGET_COLOR        = "white"
BUFFER_SIZE         = 64
KNOWN_DIAMETER_CM   = 4.0       # ping pong ball
FOCAL_LENGTH_PX     = 433.0     # replace after calibration
KNOWN_GOAL_WIDTH_CM = 30.0      # measure your real goal width in cm
SHOW_MASK           = True      # Check what's picked up
KERNEL_SIZE         = 5         # Used for morphology.
goal_center         = None
MIN_AREA            = 5000      # If the goal is further, lower this. Probably could fuck up something though.
MAX_AREA            = 60000     # If goal isn't detected much, lower this as well.
POLY_EPSILON        = 0.04      # approxPolyDP tolerance (fraction of perimeter) higher = looser shape matching
ASPECT_RATIO_MIN    = 0.3       # width/height — rectangles are wider than tall
ASPECT_RATIO_MAX    = 1.5       # upper bound to reject tall thin blobs
goal_history        = deque(maxlen=8)

# ── Color ranges (HSV) ──────────────────────────────────
color_ranges = {
    "green": ((29, 86, 6),   (64, 255, 255)),
    "blue":  ((90, 80, 50),  (130, 255, 255)),
    "white": ((0, 0, 168),   (172, 111, 255)),
}
red_lower1, red_upper1 = (0,   120, 70), (10,  255, 255)
red_lower2, red_upper2 = (170, 120, 70), (180, 255, 255)

# FOR GOAL BOARDER. Change to whatever color you need.
green_lower = (35, 80, 80)
green_upper = (85, 255, 255)

# ── Calibration mode ──
CALIBRATION_MODE  = False
KNOWN_DISTANCE_CM = 20.0

pts = deque(maxlen=BUFFER_SIZE)
cap = cv2.VideoCapture(0)               # COMMENT THIS.
# cap = cv2.VideoCapture(STREAM_URL)

def draw_trajectory(frame, ball_pt, goal_pt):
    bx, by = ball_pt
    gx, gy = goal_pt

    # Dashed line
    dist   = np.hypot(gx - bx, gy - by)                                 # straight-line pixel distance between ball and goal.
    n_segs = max(1, int(dist / 15))                                     # split that distance into segments of ~15px each.
    for i in range(n_segs):
        if i % 2 == 0:                                                  # only draw every OTHER segment, creates the dashes.
            t0 = i / n_segs                                             # start of this segment (0.0 to 1.0 along the line)
            t1 = (i + 1) / n_segs                                       # end of this segment
            p0 = (int(bx + t0 * (gx - bx)), int(by + t0 * (gy - by)))   # start point in pixels
            p1 = (int(bx + t1 * (gx - bx)), int(by + t1 * (gy - by)))   # end point in pixels
            cv2.line(frame, p0, p1, (0, 255, 0), 2, cv2.LINE_AA)        # draw the dash

    cv2.arrowedLine(frame, ball_pt, goal_pt, (0, 255, 0), 2,
                    cv2.LINE_AA, tipLength=0.08)

    # Robot approach position (25% along the line from ball toward goal)
    rx = int(bx + 0.25 * (gx - bx))
    ry = int(by + 0.25 * (gy - by))

    cv2.circle(frame, (rx, ry), 8, (0, 165, 255), -1)
    cv2.circle(frame, (rx, ry), 12, (255, 255, 255), 2)

    cv2.putText(frame, "Robot pos", (rx + 14, ry + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # ── Screen-space alignment / steering angle ─────────
    # Robot camera forward direction is the CENTER of the image.
    # We compare target X position against image center.
    frame_center_x = frame.shape[1] // 2

    # Horizontal error from screen center.
    dx = rx - frame_center_x

    # Vertical distance from bottom of image.
    # Bottom ≈ closer to robot.
    dy = frame.shape[0] - ry

    # Steering angle relative to robot forward direction.
    # 0°   = straight ahead
    # +deg = target is to the RIGHT
    # -deg = target is to the LEFT
    angle_deg = np.degrees(np.arctan2(dx, dy))

    # Draw camera center line for debugging.
    cv2.line(frame,
             (frame_center_x, 0),
             (frame_center_x, frame.shape[0]),
             (255, 255, 255), 1)

    # Display steering angle.
    cv2.putText(frame,
                f"Steer: {angle_deg:.1f} deg",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    # Return values for robot movement logic.
    return angle_deg, (rx, ry)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.open(STREAM_URL)
        continue

    frame = cv2.resize(frame, (600, 400))

    # ── Ball Detection ───────────────────────────────────
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    if TARGET_COLOR == "red":
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, red_lower1, red_upper1),
            cv2.inRange(hsv, red_lower2, red_upper2)
        )
    else:
        lower, upper = color_ranges[TARGET_COLOR]
        mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.erode(mask,  None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            pixel_diameter = radius * 2

            if CALIBRATION_MODE:
                focal_length = (pixel_diameter * KNOWN_DISTANCE_CM) / KNOWN_DIAMETER_CM
                print(f"Pixel diameter: {pixel_diameter:.1f} | Focal length: {focal_length:.2f}")
                cv2.putText(frame, f"FL: {focal_length:.1f} px  |  Hold ball at {KNOWN_DISTANCE_CM}cm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                distance_cm = (KNOWN_DIAMETER_CM * FOCAL_LENGTH_PX) / pixel_diameter

                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.rectangle(frame,
                              (int(x - radius), int(y - radius)),
                              (int(x + radius), int(y + radius)),
                              (255, 0, 0), 2)
                cv2.putText(frame, f"Ball dist: {distance_cm:.1f} cm",
                            (int(x) - 60, int(y) - int(radius) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    pts.appendleft(center)

    # ── Trail (ball history) ─────────────────────────────
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = max(1, int(np.sqrt(BUFFER_SIZE / float(i + 1)) * 2.5))
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # ── Goal Detection ───────────────────────────────────
    # Isolate Target Color.
    blurred_goal = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv_goal     = cv2.cvtColor(blurred_goal, cv2.COLOR_BGR2HSV)
    mask_goal    = cv2.inRange(hsv_goal, green_lower, green_upper)
    kernel_goal  = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask_goal    = cv2.morphologyEx(mask_goal, cv2.MORPH_OPEN,  kernel_goal)
    mask_goal    = cv2.morphologyEx(mask_goal, cv2.MORPH_CLOSE, kernel_goal)

    # Find Contours.
    cnts_goal, _ = cv2.findContours(mask_goal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    goal_center  = None
    best_contour = None
    best_score   = 0  # Largest valid rectangle

    for c_goal in cnts_goal:
        area = cv2.contourArea(c_goal)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        xg, yg, wg, hg = cv2.boundingRect(c_goal)
        aspect = wg / float(hg)

        if not (ASPECT_RATIO_MIN <= aspect <= ASPECT_RATIO_MAX):
            continue

        # Solidity: how much of the bounding box is filled.
        # A U-shape will have low solidity. (~0.4-0.6)
        # A solid rectangle will have high solidity. (~0.8+)
        solidity = area / float(wg * hg)
        if not (0.4 <= solidity <= 0.8):  # tune these if the shape isn't detected well.
            continue

        perimeter = cv2.arcLength(c_goal, True)
        approx    = cv2.approxPolyDP(c_goal, POLY_EPSILON * perimeter, True)

        if area > best_score:
            best_score   = area
            best_contour = approx
            goal_center  = (xg + wg // 2, yg + hg // 2)

    if goal_center is not None:
        goal_history.append(goal_center)
        goal_center = (
            int(np.mean([p[0] for p in goal_history])),
            int(np.mean([p[1] for p in goal_history]))
        )

    if best_contour is not None and goal_center is not None:
        # Outline Detected Rectangle/U Shape.
        cv2.drawContours(frame, [best_contour], -1, (255, 100, 0), 3)

        # Center dot.
        cv2.circle(frame, goal_center, 6, (255, 100, 0), -1)

        # Label.
        cv2.putText(frame, "GOAL",
                    (goal_center[0] - 20, goal_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 0), 2)

        # Goal distance using apparent width (same focal length formula as ball).
        xg, yg, wg, hg = cv2.boundingRect(best_contour)
        if wg > 0:
            goal_dist_cm = (KNOWN_GOAL_WIDTH_CM * FOCAL_LENGTH_PX) / wg
            cv2.putText(frame, f"Goal dist: {goal_dist_cm:.1f} cm",
                        (goal_center[0] - 60, goal_center[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)

    else:
        # Slowly fade history when goal is lost so center doesn't freeze forever.
        if len(goal_history) > 0:
            goal_history.popleft()
        cv2.putText(frame, "No goal marker found",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)

    # ── Trajectory ───────────────────────────────────────
    if center is not None and goal_center is not None:

        # Draw path + compute steering info.
        angle_deg, robot_target = draw_trajectory(frame, center, goal_center)

        # ── Robot Movement Logic ─────────────────────────
        # These thresholds prevent tiny steering corrections.
        if angle_deg > 10:
            turn_right()        # EDIT THESE.

        elif angle_deg < -10:
            turn_left()         # EDIT THESE.

        else:
            forward()           # EDIT THESE.

    # ── Display ──────────────────────────────────────────
    cv2.imshow("Frame", frame)
    cv2.imshow("Ball Mask", mask)
    if SHOW_MASK:
        cv2.imshow("Goal Mask", mask_goal)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()