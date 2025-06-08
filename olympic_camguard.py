#!/usr/bin/env python3
"""
Olympic CamGuard - 場館 AI 拍攝監控系統
最小可執行版本 (Alpha 測試階段)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel
import uvicorn
from geopy.distance import geodesic
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Olympic CamGuard API",
    description="場館 AI 拍攝監控系統 - Alpha 版本",
    version="0.1.0"
)

# 數據模型


class LocationData(BaseModel):
    latitude: float
    longitude: float
    timestamp: str
    device_id: str


class CameraStatus(BaseModel):
    is_blocked: bool
    reason: str
    location: Optional[LocationData] = None


class AIAnalysisResult(BaseModel):
    is_venue_detected: bool
    confidence: float
    detected_objects: List[str]
    risk_level: str  # "low", "medium", "high"


# 模擬奧運場館座標 (以台北為例)
VENUE_LOCATIONS = {
    "main_stadium": (25.1330, 121.5654),  # 台北主場館
    "aquatic_center": (25.1320, 121.5644),  # 水上運動中心
    "gymnasium": (25.1340, 121.5664),  # 體育館
    "nccu": (25.1340, 121.5277),  # 國立政治大學
}

# 地理圍欄半徑 (公尺)
GEOFENCE_RADIUS = 500  # 500公尺保護區域
WARNING_RADIUS = 1000  # 1000公尺警告區域

# 全域變量儲存狀態
camera_blocked_devices = set()
detection_logs = []


def check_geofence(lat: float, lon: float) -> Tuple[bool, str, float]:
    """
    檢查GPS座標是否在地理圍欄內

    Returns:
        (is_in_restricted_zone, zone_name, distance)
    """
    user_location = (lat, lon)

    for venue_name, venue_coords in VENUE_LOCATIONS.items():
        distance = geodesic(user_location, venue_coords).meters

        if distance <= GEOFENCE_RADIUS:
            return True, venue_name, distance
        elif distance <= WARNING_RADIUS:
            return False, f"{venue_name}_warning", distance

    return False, "safe_zone", float('inf')


def analyze_image_for_venue(image_data: bytes) -> AIAnalysisResult:
    """
    使用 OpenCV 進行基礎圖像分析 (模擬 AI 檢測)
    在真實場景中，這裡會是 YOLOv8 或其他 AI 模型
    """
    try:
        # 轉換圖像
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("無法解析圖像")

        # 模擬場館檢測邏輯 (基於顏色、形狀等特徵)
        detected_objects = []
        confidence = 0.0

        # 檢測大型建築物結構 (模擬)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_structures = [c for c in contours if cv2.contourArea(c) > 10000]

        if large_structures:
            detected_objects.append("large_building")
            confidence += 0.3

        # 檢測奧運相關顏色 (藍、黃、黑、綠、紅)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 藍色範圍
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        if cv2.countNonZero(blue_mask) > img.shape[0] * img.shape[1] * 0.1:
            detected_objects.append("olympic_colors")
            confidence += 0.2

        # 檢測圓形結構 (可能是體育場)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            detected_objects.append("stadium_structure")
            confidence += 0.4

        # 檢測文字 (簡化版)
        if len(detected_objects) >= 2:
            detected_objects.append("venue_signage")
            confidence += 0.3

        # 判斷是否檢測到場館
        is_venue_detected = confidence > 0.5

        # 風險等級評估
        if confidence > 0.8:
            risk_level = "high"
        elif confidence > 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"

        return AIAnalysisResult(
            is_venue_detected=is_venue_detected,
            confidence=min(confidence, 1.0),
            detected_objects=detected_objects,
            risk_level=risk_level
        )

    except Exception as e:
        logger.error(f"圖像分析錯誤: {str(e)}")
        return AIAnalysisResult(
            is_venue_detected=False,
            confidence=0.0,
            detected_objects=[],
            risk_level="low"
        )


@app.get("/", response_class=HTMLResponse)
async def home():
    """返回測試用的 HTML 界面"""
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Olympic CamGuard - Alpha 測試</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .safe { background-color: #d4edda; color: #155724; }
            .warning { background-color: #fff3cd; color: #856404; }
            .danger { background-color: #f8d7da; color: #721c24; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background-color: #0056b3; }
            input[type="file"] { margin: 10px 0; }
            .result { margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏅 Olympic CamGuard</h1>
                <h3>場館 AI 拍攝監控系統 - Alpha 測試版</h3>
            </div>
            
            <div class="section">
                <h3>📍 GPS 定位測試</h3>
                <button onclick="testLocation()">獲取當前位置</button>
                <button onclick="simulateVenueLocation()">模擬場館內位置</button>
                <button onclick="simulateWarningLocation()">模擬警告區域</button>
                <div id="locationResult" class="result"></div>
            </div>
            
            <div class="section">
                <h3>📷 相機控制測試</h3>
                <button onclick="checkCameraStatus()">檢查相機狀態</button>
                <button onclick="simulatePhotoCapture()">模擬拍照</button>
                <div id="cameraResult" class="result"></div>
            </div>
            
            <div class="section">
                <h3>🧠 AI 圖像分析測試</h3>
                <input type="file" id="imageInput" accept="image/*">
                <button onclick="analyzeImage()">分析圖像</button>
                <div id="imageResult" class="result"></div>
            </div>
            
            <div class="section">
                <h3>📊 系統狀態</h3>
                <button onclick="getSystemStatus()">刷新狀態</button>
                <div id="systemStatus" class="result"></div>
            </div>
        </div>

        <script>
            let currentDeviceId = 'test-device-' + Math.random().toString(36).substr(2, 9);
            
            async function testLocation() {
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(async (position) => {
                        const locationData = {
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude,
                            timestamp: new Date().toISOString(),
                            device_id: currentDeviceId
                        };
                        
                        try {
                            const response = await fetch('/api/check-location', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(locationData)
                            });
                            const result = await response.json();
                            displayLocationResult(result);
                        } catch (error) {
                            document.getElementById('locationResult').innerHTML = 
                                '<div class="status danger">位置檢查失敗: ' + error.message + '</div>';
                        }
                    }, (error) => {
                        document.getElementById('locationResult').innerHTML = 
                            '<div class="status danger">無法獲取位置: ' + error.message + '</div>';
                    });
                } else {
                    document.getElementById('locationResult').innerHTML = 
                        '<div class="status danger">瀏覽器不支援地理定位</div>';
                }
            }
            
            async function simulateVenueLocation() {
                const locationData = {
                    latitude: 25.0330,
                    longitude: 121.5654,
                    timestamp: new Date().toISOString(),
                    device_id: currentDeviceId
                };
                
                const response = await fetch('/api/check-location', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(locationData)
                });
                const result = await response.json();
                displayLocationResult(result);
            }
            
            async function simulateWarningLocation() {
                const locationData = {
                    latitude: 25.0350,
                    longitude: 121.5680,
                    timestamp: new Date().toISOString(),
                    device_id: currentDeviceId
                };
                
                const response = await fetch('/api/check-location', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(locationData)
                });
                const result = await response.json();
                displayLocationResult(result);
            }
            
            function displayLocationResult(result) {
                let statusClass = 'safe';
                if (result.risk_level === 'high') statusClass = 'danger';
                else if (result.risk_level === 'medium') statusClass = 'warning';
                
                document.getElementById('locationResult').innerHTML = 
                    '<div class="status ' + statusClass + '">' +
                    '<strong>位置狀態:</strong> ' + result.message + '<br>' +
                    '<strong>距離最近場館:</strong> ' + Math.round(result.distance) + ' 公尺<br>' +
                    '<strong>風險等級:</strong> ' + result.risk_level +
                    '</div>';
            }
            
            async function checkCameraStatus() {
                try {
                    const response = await fetch('/api/camera-status/' + currentDeviceId);
                    const result = await response.json();
                    
                    let statusClass = result.is_blocked ? 'danger' : 'safe';
                    let statusText = result.is_blocked ? '🚫 已封鎖' : '✅ 可使用';
                    
                    document.getElementById('cameraResult').innerHTML = 
                        '<div class="status ' + statusClass + '">' +
                        '<strong>相機狀態:</strong> ' + statusText + '<br>' +
                        '<strong>原因:</strong> ' + result.reason +
                        '</div>';
                } catch (error) {
                    document.getElementById('cameraResult').innerHTML = 
                        '<div class="status danger">狀態檢查失敗: ' + error.message + '</div>';
                }
            }
            
            async function simulatePhotoCapture() {
                try {
                    const response = await fetch('/api/capture-photo', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ device_id: currentDeviceId })
                    });
                    const result = await response.json();
                    
                    let statusClass = result.allowed ? 'safe' : 'danger';
                    
                    document.getElementById('cameraResult').innerHTML = 
                        '<div class="status ' + statusClass + '">' +
                        '<strong>拍照請求:</strong> ' + (result.allowed ? '允許' : '拒絕') + '<br>' +
                        '<strong>訊息:</strong> ' + result.message +
                        '</div>';
                } catch (error) {
                    document.getElementById('cameraResult').innerHTML = 
                        '<div class="status danger">拍照測試失敗: ' + error.message + '</div>';
                }
            }
            
            async function analyzeImage() {
                const fileInput = document.getElementById('imageInput');
                if (!fileInput.files[0]) {
                    alert('請選擇一張圖片');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    const response = await fetch('/api/analyze-image', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    
                    let statusClass = 'safe';
                    if (result.risk_level === 'high') statusClass = 'danger';
                    else if (result.risk_level === 'medium') statusClass = 'warning';
                    
                    document.getElementById('imageResult').innerHTML = 
                        '<div class="status ' + statusClass + '">' +
                        '<strong>場館檢測:</strong> ' + (result.is_venue_detected ? '是' : '否') + '<br>' +
                        '<strong>信心度:</strong> ' + Math.round(result.confidence * 100) + '%<br>' +
                        '<strong>檢測物件:</strong> ' + result.detected_objects.join(', ') + '<br>' +
                        '<strong>風險等級:</strong> ' + result.risk_level +
                        '</div>';
                } catch (error) {
                    document.getElementById('imageResult').innerHTML = 
                        '<div class="status danger">圖像分析失敗: ' + error.message + '</div>';
                }
            }
            
            async function getSystemStatus() {
                try {
                    const response = await fetch('/api/system-status');
                    const result = await response.json();
                    
                    document.getElementById('systemStatus').innerHTML = 
                        '<div class="status safe">' +
                        '<strong>系統運行時間:</strong> ' + result.uptime + '<br>' +
                        '<strong>已封鎖設備數:</strong> ' + result.blocked_devices_count + '<br>' +
                        '<strong>檢測記錄數:</strong> ' + result.detection_logs_count + '<br>' +
                        '<strong>活躍場館:</strong> ' + result.active_venues.join(', ') +
                        '</div>';
                } catch (error) {
                    document.getElementById('systemStatus').innerHTML = 
                        '<div class="status danger">狀態獲取失敗: ' + error.message + '</div>';
                }
            }
            
            // 頁面載入時初始化
            window.onload = function() {
                getSystemStatus();
            };
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)


@app.post("/api/check-location")
async def check_location(location: LocationData):
    """檢查GPS位置並判斷是否在限制區域"""
    is_restricted, zone_name, distance = check_geofence(
        location.latitude,
        location.longitude
    )

    risk_level = "low"
    message = "位置安全"

    if is_restricted:
        risk_level = "high"
        message = f"您在 {zone_name} 保護區域內，禁止拍攝"
        camera_blocked_devices.add(location.device_id)
    elif "warning" in zone_name:
        risk_level = "medium"
        message = f"您接近 {zone_name.replace('_warning', '')} 場館，請注意拍攝限制"
    else:
        message = "位置安全，可正常使用相機"
        camera_blocked_devices.discard(location.device_id)

    # 記錄檢測
    detection_logs.append({
        "timestamp": location.timestamp,
        "device_id": location.device_id,
        "location": (location.latitude, location.longitude),
        "zone": zone_name,
        "distance": distance,
        "risk_level": risk_level
    })

    return {
        "message": message,
        "risk_level": risk_level,
        "zone": zone_name,
        "distance": distance,
        "camera_blocked": is_restricted
    }


@app.get("/api/camera-status/{device_id}")
async def get_camera_status(device_id: str):
    """獲取特定設備的相機狀態"""
    is_blocked = device_id in camera_blocked_devices
    reason = "位置限制" if is_blocked else "正常使用"

    return CameraStatus(
        is_blocked=is_blocked,
        reason=reason,
        location=None
    )


@app.post("/api/capture-photo")
async def capture_photo(request: dict):
    """模擬拍照請求"""
    device_id = request.get("device_id")

    if device_id in camera_blocked_devices:
        return {
            "allowed": False,
            "message": "拍攝被阻止：您目前位於限制區域內",
            "device_id": device_id
        }

    return {
        "allowed": True,
        "message": "拍攝許可：位置安全",
        "device_id": device_id
    }


@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """分析上傳的圖像是否包含場館內容"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖像文件")

    try:
        image_data = await file.read()
        result = analyze_image_for_venue(image_data)

        # 記錄分析結果
        detection_logs.append({
            "timestamp": datetime.now().isoformat(),
            "type": "image_analysis",
            "filename": file.filename,
            "result": result.dict(),
        })

        return result

    except Exception as e:
        logger.error(f"圖像分析錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"圖像分析失敗: {str(e)}")


@app.get("/api/system-status")
async def get_system_status():
    """獲取系統狀態"""
    return {
        "uptime": "運行中",
        "blocked_devices_count": len(camera_blocked_devices),
        "detection_logs_count": len(detection_logs),
        "active_venues": list(VENUE_LOCATIONS.keys()),
        "geofence_radius": GEOFENCE_RADIUS,
        "warning_radius": WARNING_RADIUS
    }


@app.get("/api/detection-logs")
async def get_detection_logs(limit: int = 50):
    """獲取檢測記錄"""
    return {
        "logs": detection_logs[-limit:],
        "total": len(detection_logs)
    }

if __name__ == "__main__":
    print("🏅 Olympic CamGuard - Alpha 版本啟動中...")
    print("📍 地理圍欄已設置")
    print("🧠 AI 分析模組已載入")
    print("📱 API 服務器運行於: http://localhost:8000")
    print("\n按 Ctrl+C 停止系統")
    print("=" * 50)

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\n👋 系統已停止")
