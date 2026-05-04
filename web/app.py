"""
web/app.py — Surveillance Platform with LBPH recognition + DCT watermark.
Run: python web/app.py → http://localhost:5000
"""
import sys, os, cv2, time, threading, base64
import numpy as np
from datetime import datetime
from flask import (Flask, Response, render_template, jsonify,
                   request, session, redirect, url_for, send_from_directory)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (CAMERA_INDEX, CAMERA_ID, CAPTURES_AUTH, CAPTURES_UNAUTH,
                    CAPTURES_UNK, SEED_PERSONS, ALERT_COLORS)
from core.utils import resize_frame, FPSCounter, ensure_dirs, safe_filename
from modules.recognition.detector   import FaceDetector
from modules.recognition.recognizer import FaceRecognizer
from modules.watermark.watermark    import watermark_capture, compute_sha256
from modules.database.db            import (init_db, insert_person_if_missing,
                                            insert_event, get_recent_events)
from modules.verification.verify    import verify_capture

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)

SESSION_DURATION = 300
WARN_AT          = 30
SAVE_DIR_MAP     = {"AUTHORIZED":CAPTURES_AUTH,"UNAUTHORIZED":CAPTURES_UNAUTH,"UNKNOWN":CAPTURES_UNK}

_frame_lock   = threading.Lock()
_latest_frame = None
_state_lock   = threading.Lock()
_state = {"person":"—","category":"—","alert_level":"—","confidence":0.0,
          "fps":0.0,"last_capture":"—","uptime_start":time.time()}

_detector = _recognizer = None

def _models():
    global _detector, _recognizer
    if _detector is None:
        _detector   = FaceDetector()
        _recognizer = FaceRecognizer()
    return _detector, _recognizer

def _camera_thread():
    global _latest_frame
    fps_ctr = FPSCounter()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Camera] ERROR: cannot open camera {CAMERA_INDEX}"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[Camera] Running on index {CAMERA_INDEX}")
    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.05); continue
        new_fps = fps_ctr.tick()
        if new_fps:
            with _state_lock: _state["fps"] = round(new_fps, 1)
        with _frame_lock: _latest_frame = frame.copy()
        time.sleep(0.03)

def _get_frame():
    with _frame_lock:
        return _latest_frame.copy() if _latest_frame is not None else None

def _scan_face():
    frame = _get_frame()
    if frame is None: return {"success":False,"error":"Camera not ready"}

    detector, recognizer = _models()
    frame   = resize_frame(frame)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Use equalised gray for DETECTION, raw gray for RECOGNITION
    gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq  = cv2.equalizeHist(gray_raw)
    boxes    = detector.detect(gray_eq)

    if not boxes: return {"success":False,"error":"No face detected. Please try again."}

    x,y,w,h  = max(boxes, key=lambda b: b[2]*b[3])
    roi_gray  = gray_raw[y:y+h, x:x+w]   # raw gray for LBPH
    name, category, alert_level, confidence = recognizer.predict(roi_gray)

    color = ALERT_COLORS.get(alert_level, (200,200,200))
    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
    cv2.putText(frame,f"{name}  [{confidence:.0f}]",(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)

    metadata = {"timestamp":now_str,"camera_id":CAMERA_ID,"person_name":name,
                "category":category,"alert_level":alert_level}
    watermarked = watermark_capture(frame, metadata)
    save_dir    = SAVE_DIR_MAP.get(category, CAPTURES_UNK)
    filename    = safe_filename(name, now_str)
    img_path    = os.path.normpath(os.path.join(save_dir, filename))

    cv2.imwrite(img_path, watermarked, [cv2.IMWRITE_JPEG_QUALITY, 95])
    sha = compute_sha256(img_path)
    insert_event(name, category, CAMERA_ID, now_str, img_path, sha, alert_level)

    _, buf = cv2.imencode(".jpg", watermarked, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64    = base64.b64encode(buf).decode("utf-8")

    with _state_lock:
        _state.update({"person":name,"category":category,"alert_level":alert_level,
                       "confidence":round(float(confidence),1),"last_capture":filename})

    print(f"  [SCAN] {alert_level:<8} | {name:<15} | conf={confidence:.1f}%")
    return {"success":True,"name":name,"category":category,"alert_level":alert_level,
            "confidence":round(float(confidence),1),"timestamp":now_str,
            "filename":filename,"image_b64":b64}

def _gen_login():
    cascade = cv2.CascadeClassifier(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "cascades","haarcascade_frontalface_alt2.xml"))
    while True:
        frame = _get_frame()
        if frame is None:
            frame = np.zeros((480,640,3),dtype=np.uint8)
            cv2.putText(frame,"Initialising...",(190,240),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,180,80),2)
        else:
            frame  = resize_frame(frame)
            h, w   = frame.shape[:2]
            cv2.ellipse(frame,(w//2,h//2),(120,155),0,0,360,(0,200,100),2)
            cv2.putText(frame,"Position your face within the oval",
                        (w//2-140,h//2+185),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,100),1)
        _, buf = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,70])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
        time.sleep(0.04)

def _gen_dashboard():
    while True:
        frame = _get_frame()
        if frame is None: time.sleep(0.04); continue
        detector, recognizer = _models()
        frame    = resize_frame(frame)
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq  = cv2.equalizeHist(gray_raw)
        for (x,y,w,h) in detector.detect(gray_eq):
            roi  = gray_raw[y:y+h, x:x+w]
            name,cat,alert,conf = recognizer.predict(roi)
            color = ALERT_COLORS.get(alert,(200,200,200))
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            label = f"{name}  [{conf:.0f}]" if conf > 0 else name
            cv2.putText(frame,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,color,1,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y+h-20),(x+w,y+h),color,-1)
            cv2.putText(frame,cat,(x+4,y+h-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,0,0),1,cv2.LINE_AA)
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame,f"CAM-01  |  {ts}",(8,frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,200,100),1,cv2.LINE_AA)
        _, buf = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,75])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
        time.sleep(0.033)

def _is_auth():    return time.time() < session.get("expires_at",0)
def _remaining():  return max(0,int(session.get("expires_at",0)-time.time()))
def _create_session(name):
    session["user"]       = name
    session["expires_at"] = time.time()+SESSION_DURATION
    session["login_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def _extend_session(): session["expires_at"] = time.time()+SESSION_DURATION

@app.route("/")
def index():
    return redirect(url_for("dashboard")) if _is_auth() else render_template("login.html")

@app.route("/video_feed_login")
def video_feed_login():
    return Response(_gen_login(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_dashboard")
def video_feed_dashboard():
    if not _is_auth(): return "",403
    return Response(_gen_dashboard(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/scan",methods=["POST"])
def api_scan():
    result = _scan_face()
    if result["success"]:
        if result["category"]=="AUTHORIZED":
            _create_session(result["name"]); result["action"]="login"
        elif result["category"]=="UNAUTHORIZED":
            result["action"]="denied"
        else:
            result["action"]="alarm"
    return jsonify(result)

@app.route("/dashboard")
def dashboard():
    if not _is_auth(): return redirect(url_for("index"))
    return render_template("dashboard.html",user=session.get("user","—"),
                           login_time=session.get("login_time","—"))

@app.route("/access_denied")
def access_denied():
    return render_template("access_denied.html")

@app.route("/alarm")
def alarm():
    return render_template("alarm.html")

@app.route("/api/session")
def api_session_status():
    if not _is_auth(): return jsonify({"authenticated":False,"remaining":0})
    r = _remaining()
    return jsonify({"authenticated":True,"remaining":r,"warning":r<=WARN_AT,
                    "user":session.get("user","—")})

@app.route("/api/rescan",methods=["POST"])
def api_rescan():
    result = _scan_face()
    if not result["success"]: return jsonify(result)
    if result["category"]=="AUTHORIZED":
        _extend_session(); result["action"]="extended"
    else:
        session.clear(); result["action"]="logout"
    return jsonify(result)

@app.route("/api/logout",methods=["POST"])
def api_logout():
    session.clear(); return jsonify({"success":True})

@app.route("/api/status")
def api_status():
    with _state_lock: s=dict(_state)
    u=int(time.time()-s["uptime_start"]); h,r=divmod(u,3600); m,sec=divmod(r,60)
    s["uptime"]=f"{h:02d}:{m:02d}:{sec:02d}"; s["time"]=datetime.now().strftime("%H:%M:%S")
    return jsonify(s)

@app.route("/api/events")
def api_events():
    rows=get_recent_events(limit=20)
    return jsonify([{"id":r["id"],"person_name":r["person_name"],"category":r["category"],
                     "alert_level":r["alert_level"],"timestamp":r["timestamp"],
                     "image_path":os.path.basename(r["image_path"])} for r in rows])

@app.route("/api/stats")
def api_stats():
    rows=get_recent_events(limit=1000)
    counts={"AUTHORIZED":0,"UNAUTHORIZED":0,"UNKNOWN":0}
    for r in rows:
        if r["category"] in counts: counts[r["category"]]+=1
    return jsonify(counts)

@app.route("/api/verify",methods=["POST"])
def api_verify():
    filename=(request.get_json() or {}).get("filename","")
    for sub in [CAPTURES_AUTH,CAPTURES_UNAUTH,CAPTURES_UNK]:
        p=os.path.join(sub,filename)
        if os.path.isfile(p): return jsonify({"valid":verify_capture(p,silent=True)})
    return jsonify({"valid":False,"error":"Not found"}),404

@app.route("/captures/<path:filename>")
def serve_capture(filename):
    for sub in [CAPTURES_AUTH,CAPTURES_UNAUTH,CAPTURES_UNK]:
        p=os.path.join(sub,filename)
        if os.path.isfile(p): return send_from_directory(sub,filename)
    return "Not found",404

if __name__=="__main__":
    init_db()
    for p in SEED_PERSONS: insert_person_if_missing(p["name"],p["status"])
    ensure_dirs(CAPTURES_AUTH,CAPTURES_UNAUTH,CAPTURES_UNK)
    threading.Thread(target=_camera_thread,daemon=True).start()
    print("\n"+"="*50+"\n  Smart Surveillance Platform\n  http://localhost:5000\n"+"="*50+"\n")
    app.run(host="0.0.0.0",port=5000,debug=False,threaded=True)
