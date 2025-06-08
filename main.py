#!/usr/bin/env python3
"""
Olympic CamGuard - 場館 AI 拍攝監控系統
最小可執行版本 (Alpha 測試階段) - 增強 AI 分析版本
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
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Olympic CamGuard API",
    description="場館 AI 拍攝監控系統 - Alpha 版本 (Enhanced AI)",
    version="0.2.0"
)

# 初始化線程池執行器
executor = ThreadPoolExecutor(max_workers=6)

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
    detailed_analysis: Optional[Dict] = None


# 模擬奧運場館座標 (以台北為例)
VENUE_LOCATIONS = {
    "main_stadium": (25.04264080324845, 121.55948513642994),  # 台北大巨蛋
    "aquatic_center": (25.049884269960565, 121.55214558983418),  # 台北田徑場
    "gymnasium": (25.067716551819338, 121.59740660517531),  # 台北網球中心
    "head_quarters": (25.030327766233093, 121.52811482051422),  # NCCU

}

# 地理圍欄半徑 (公尺)
GEOFENCE_RADIUS = 500  # 500公尺保護區域
WARNING_RADIUS = 1000  # 1000公尺警告區域

# 全域變量儲存狀態
camera_blocked_devices = set()
detection_logs = []

# === 增強的 AI 檢測模組 ===


class EnhancedVenueDetector:
    """增強版場館檢測器"""

    def __init__(self):
        # 奧運相關顏色定義 (HSV)
        self.olympic_colors = {
            "blue": [(100, 150, 50), (130, 255, 255)],
            "yellow": [(20, 100, 100), (30, 255, 255)],
            "green": [(50, 100, 50), (70, 255, 255)],
            "red": [(0, 120, 70), (10, 255, 255)],
            "black": [(0, 0, 0), (180, 255, 30)]
        }

        # 檢測器權重配置
        self.detection_weights = {
            "olympic_rings": 2.0,
            "stadium_structure": 1.8,
            "sports_equipment": 1.5,
            "architectural_features": 1.2,
            "olympic_colors": 1.0,
            "venue_signage": 1.0,
            "crowd_patterns": 0.8,
            "geometric_patterns": 0.7
        }

    def preprocess_image(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """多層次圖像預處理"""
        processed = {}

        # 基礎處理
        processed['original'] = img.copy()
        processed['gray'] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed['hsv'] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 增強處理
        processed['enhanced'] = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        processed['denoised'] = cv2.bilateralFilter(img, 9, 75, 75)

        # 多尺度邊緣檢測
        processed['edges_low'] = cv2.Canny(processed['gray'], 30, 100)
        processed['edges_high'] = cv2.Canny(processed['gray'], 80, 200)
        processed['edges_combined'] = cv2.bitwise_or(
            processed['edges_low'], processed['edges_high']
        )

        # 自適應閾值
        processed['adaptive'] = cv2.adaptiveThreshold(
            processed['gray'], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return processed

    def detect_olympic_rings(self, img_data: Dict) -> Tuple[float, List[str]]:
        """奧運五環檢測"""
        gray = img_data['gray']
        confidence = 0.0
        features = []

        # 多參數圓形檢測
        circles_sets = [
            cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30,
                             param1=50, param2=30, minRadius=15, maxRadius=80),
            cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                             param1=40, param2=25, minRadius=10, maxRadius=100)
        ]

        all_circles = []
        for circles in circles_sets:
            if circles is not None:
                all_circles.extend(np.round(circles[0, :]).astype("int"))

        if all_circles:
            # 去重複
            unique_circles = self._remove_duplicate_circles(all_circles)
            ring_count = len(unique_circles)

            features.append(f"{ring_count}個圓形結構")

            # 檢查顏色分佈
            ring_colors = self._analyze_circle_colors(
                img_data['hsv'], unique_circles)
            olympic_color_match = sum(1 for color in ring_colors
                                      if color in self.olympic_colors.keys())

            if ring_count >= 3:
                confidence += 0.4
                if ring_count >= 5:
                    confidence += 0.2
                    features.append("多圓環結構")

            if olympic_color_match >= 2:
                confidence += 0.3
                features.append(f"奧運色彩匹配({olympic_color_match})")

        return min(confidence, 0.9), features

    def detect_stadium_structure(self, img_data: Dict) -> Tuple[float, List[str]]:
        """體育場結構檢測"""
        edges = img_data['edges_combined']
        confidence = 0.0
        features = []

        # 檢測大型輪廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > 3000]

        if large_contours:
            # 分析結構特徵
            stadium_features = 0
            for contour in large_contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    # 檢查是否為體育場典型形狀
                    if 0.3 < circularity < 0.9 and area > 10000:
                        stadium_features += 1

            if stadium_features > 0:
                confidence += min(stadium_features * 0.25, 0.6)
                features.append(f"{stadium_features}個體育場結構")

        # 檢測看台階梯
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40,
                                minLineLength=50, maxLineGap=10)
        if lines is not None:
            parallel_groups = self._find_parallel_line_groups(lines)
            if len(parallel_groups) >= 3:
                confidence += 0.3
                features.append("看台階梯結構")

        return min(confidence, 0.8), features

    def detect_sports_equipment(self, img_data: Dict) -> Tuple[float, List[str]]:
        """體育設備檢測"""
        hsv = img_data['hsv']
        confidence = 0.0
        features = []

        # 檢測運動場白線
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))
        white_contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        long_lines = [c for c in white_contours if cv2.contourArea(c) > 300]
        if long_lines:
            confidence += 0.4
            features.append("運動場標線")

            # 檢查線條規律性
            if len(long_lines) >= 5:
                confidence += 0.2
                features.append("規律標線系統")

        # 檢測球門/設備框架
        gray = img_data['gray']
        rectangles = cv2.findContours(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY +
                          cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]

        sports_rects = []
        for rect in rectangles:
            area = cv2.contourArea(rect)
            if 1000 < area < 50000:
                approx = cv2.approxPolyDP(
                    rect, 0.02 * cv2.arcLength(rect, True), True)
                if len(approx) == 4:
                    sports_rects.append(rect)

        if sports_rects:
            confidence += min(len(sports_rects) * 0.1, 0.3)
            features.append(f"{len(sports_rects)}個運動設備")

        return min(confidence, 0.8), features

    def detect_architectural_features(self, img_data: Dict) -> Tuple[float, List[str]]:
        """建築特徵檢測"""
        edges = img_data['edges_combined']
        confidence = 0.0
        features = []

        # 檢測直線結構
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                                minLineLength=60, maxLineGap=15)

        if lines is not None:
            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                if length > 80:
                    if abs(angle) < 20 or abs(angle) > 160:
                        horizontal_lines.append(line)
                    elif 70 < abs(angle) < 110:
                        vertical_lines.append(line)

            # 評估建築規律性
            if len(horizontal_lines) > 4:
                confidence += 0.25
                features.append("水平建築線條")

            if len(vertical_lines) > 4:
                confidence += 0.25
                features.append("垂直建築線條")

            if len(horizontal_lines) > 6 and len(vertical_lines) > 6:
                confidence += 0.2
                features.append("規律建築網格")

        return min(confidence, 0.7), features

    def detect_olympic_colors_advanced(self, img_data: Dict) -> Tuple[float, List[str]]:
        """進階奧運色彩檢測"""
        hsv = img_data['hsv']
        total_pixels = hsv.shape[0] * hsv.shape[1]
        confidence = 0.0
        features = []

        detected_colors = []
        color_percentages = {}

        for color_name, (lower, upper) in self.olympic_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_pixels = cv2.countNonZero(mask)
            percentage = color_pixels / total_pixels

            if percentage > 0.03:  # 降低閾值到3%
                detected_colors.append(color_name)
                color_percentages[color_name] = percentage

        if detected_colors:
            # 基礎分數
            confidence = min(len(detected_colors) * 0.12, 0.5)
            features.append(f"檢測到: {', '.join(detected_colors)}")

            # 奧運五環完整性檢查
            olympic_set = {"blue", "yellow", "green", "red", "black"}
            detected_set = set(detected_colors)
            overlap = len(detected_set.intersection(olympic_set))

            if overlap >= 3:
                confidence += 0.25
                features.append(f"奧運色彩組合({overlap}/5)")

            if overlap >= 4:
                confidence += 0.15
                features.append("高度奧運特徵")

        return min(confidence, 0.8), features

    def detect_venue_signage(self, img_data: Dict) -> Tuple[float, List[str]]:
        """場館標識檢測"""
        gray = img_data['gray']
        confidence = 0.0
        features = []

        # MSER 文字區域檢測
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)

            text_regions = [r for r in regions if 30 < len(r) < 1500]
            if text_regions:
                confidence += min(len(text_regions) * 0.03, 0.4)
                features.append(f"{len(text_regions)}個文字區域")
        except:
            pass

        # 矩形標牌檢測
        edges = img_data['edges_combined']
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sign_rectangles = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 15000:
                approx = cv2.approxPolyDP(
                    contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    sign_rectangles += 1

        if sign_rectangles > 0:
            confidence += min(sign_rectangles * 0.08, 0.3)
            features.append(f"{sign_rectangles}個標牌結構")

        return min(confidence, 0.6), features

    def detect_crowd_patterns(self, img_data: Dict) -> Tuple[float, List[str]]:
        """人群模式檢測"""
        hsv = img_data['hsv']
        confidence = 0.0
        features = []
        total_pixels = hsv.shape[0] * hsv.shape[1]

        # 膚色檢測
        skin_lower = np.array([0, 20, 70])
        skin_upper = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        skin_ratio = cv2.countNonZero(skin_mask) / total_pixels

        if 0.01 < skin_ratio < 0.2:
            confidence += 0.2
            features.append("人群活動")

        # 色彩多樣性分析
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        non_zero_bins = np.count_nonzero(hist)
        diversity = non_zero_bins / 180.0

        if diversity > 0.4:
            confidence += 0.15
            features.append("高色彩多樣性")

        return min(confidence, 0.35), features

    def detect_geometric_patterns(self, img_data: Dict) -> Tuple[float, List[str]]:
        """幾何模式檢測"""
        gray = img_data['gray']
        confidence = 0.0
        features = []

        # 對稱性檢測
        height, width = gray.shape
        if width > 100:
            left_half = gray[:, :width//2]
            right_half = cv2.flip(gray[:, width//2:], 1)

            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]

            try:
                similarity = cv2.matchTemplate(
                    left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
                if similarity > 0.7:
                    confidence += 0.2
                    features.append("對稱結構")
            except:
                pass

        # 重複圖案檢測
        template_size = min(40, width//5, height//5)
        if template_size > 15:
            template = gray[:template_size, :template_size]
            try:
                result = cv2.matchTemplate(
                    gray, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.75)
                if len(locations[0]) > 2:
                    confidence += 0.15
                    features.append("重複圖案")
            except:
                pass

        return min(confidence, 0.35), features

    # === 輔助函數 ===

    def _remove_duplicate_circles(self, circles: List) -> List:
        """移除重複的圓形"""
        if not circles:
            return []

        unique = []
        for circle in circles:
            x, y, r = circle
            is_duplicate = False
            for ux, uy, ur in unique:
                distance = np.sqrt((x-ux)**2 + (y-uy)**2)
                if distance < max(r, ur) * 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(circle)
        return unique

    def _analyze_circle_colors(self, hsv: np.ndarray, circles: List) -> List[str]:
        """分析圓形區域的顏色"""
        colors = []
        for x, y, r in circles:
            # 創建圓形遮罩
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)

            # 分析顏色
            for color_name, (lower, upper) in self.olympic_colors.items():
                color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                overlap = cv2.bitwise_and(mask, color_mask)
                if cv2.countNonZero(overlap) > r * r * 0.1:  # 10% 重疊
                    colors.append(color_name)
                    break
        return colors

    def _find_parallel_line_groups(self, lines: np.ndarray) -> List[List]:
        """尋找平行線組"""
        if lines is None or len(lines) < 2:
            return []

        groups = []
        used = set()

        for i, line1 in enumerate(lines):
            if i in used:
                continue

            x1, y1, x2, y2 = line1[0]
            angle1 = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            group = [i]
            used.add(i)

            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used:
                    continue

                x3, y3, x4, y4 = line2[0]
                angle2 = np.arctan2(y4-y3, x4-x3) * 180 / np.pi

                if abs(angle1 - angle2) < 15:  # 平行容差
                    group.append(j)
                    used.add(j)

            if len(group) >= 2:
                groups.append(group)

        return groups


detector = EnhancedVenueDetector()


def check_geofence(lat: float, lon: float) -> Tuple[bool, str, float]:
    """
    檢查GPS座標是否在地理圍欄內

    Returns:
        (is_in_restricted_zone, zone_name, distance)
    """
    user_location = (lat, lon)
    min_distance = float('inf')
    closest_venue = "safe_zone"

    for venue_name, venue_coords in VENUE_LOCATIONS.items():
        distance = geodesic(user_location, venue_coords).meters

        if distance <= GEOFENCE_RADIUS:
            return True, venue_name, distance
        elif distance <= WARNING_RADIUS:
            return False, f"{venue_name}_warning", distance

        # 追蹤最近的場館距離
        if distance < min_distance:
            min_distance = distance
            closest_venue = venue_name

    # 返回有限的距離值，而不是無限大
    return False, "safe_zone", min_distance if min_distance != float('inf') else 999999


def analyze_image_for_venue(image_data: bytes) -> AIAnalysisResult:
    """
    增強版圖像分析函數 - 多層檢測架構
    整合多種檢測方法提高準確度
    """
    try:
        # 轉換圖像
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("無法解析圖像")

        # 預處理
        img_processed = detector.preprocess_image(img)

        # 多層檢測結果
        detection_results = {}
        all_features = []

        # === 並行檢測所有特徵 ===

        # 1. 奧運五環檢測
        conf, features = detector.detect_olympic_rings(img_processed)
        if conf > 0.05:
            detection_results["olympic_rings"] = conf
            all_features.extend(features)

        # 2. 體育場結構檢測
        conf, features = detector.detect_stadium_structure(img_processed)
        if conf > 0.05:
            detection_results["stadium_structure"] = conf
            all_features.extend(features)

        # 3. 體育設備檢測
        conf, features = detector.detect_sports_equipment(img_processed)
        if conf > 0.05:
            detection_results["sports_equipment"] = conf
            all_features.extend(features)

        # 4. 建築特徵檢測
        conf, features = detector.detect_architectural_features(img_processed)
        if conf > 0.05:
            detection_results["architectural_features"] = conf
            all_features.extend(features)

        # 5. 奧運色彩檢測 (進階版)
        conf, features = detector.detect_olympic_colors_advanced(img_processed)
        if conf > 0.05:
            detection_results["olympic_colors"] = conf
            all_features.extend(features)

        # 6. 場館標識檢測
        conf, features = detector.detect_venue_signage(img_processed)
        if conf > 0.05:
            detection_results["venue_signage"] = conf
            all_features.extend(features)

        # 7. 人群模式檢測
        conf, features = detector.detect_crowd_patterns(img_processed)
        if conf > 0.05:
            detection_results["crowd_patterns"] = conf
            all_features.extend(features)

        # 8. 幾何模式檢測
        conf, features = detector.detect_geometric_patterns(img_processed)
        if conf > 0.05:
            detection_results["geometric_patterns"] = conf
            all_features.extend(features)

        # === 加權計算最終信心度 ===
        weighted_sum = 0.0
        total_weight = 0.0

        for feature, conf in detection_results.items():
            weight = detector.detection_weights.get(feature, 1.0)
            weighted_sum += conf * weight
            total_weight += weight

        # 計算最終信心度
        if total_weight > 0:
            final_confidence = weighted_sum / total_weight
            # 添加檢測特徵數量的獎勵
            feature_bonus = min(len(detection_results) * 0.05, 0.2)
            final_confidence = min(final_confidence + feature_bonus, 1.0)
        else:
            final_confidence = 0.0

        # 判斷是否檢測到場館
        is_venue_detected = final_confidence > 0.45  # 降低閾值提高靈敏度

        # 風險等級評估 (更精確)
        if final_confidence > 0.75:
            risk_level = "high"
        elif final_confidence > 0.45:
            risk_level = "medium"
        else:
            risk_level = "low"

        # 準備詳細分析結果
        detailed_analysis = {
            "feature_scores": detection_results,
            "total_features": len(detection_results),
            "feature_details": all_features,
            "processing_info": {
                "image_size": f"{img.shape[1]}x{img.shape[0]}",
                "detection_methods": len(detector.detection_weights)
            }
        }

        return AIAnalysisResult(
            is_venue_detected=is_venue_detected,
            confidence=round(final_confidence, 3),
            detected_objects=list(detection_results.keys()),
            risk_level=risk_level,
            detailed_analysis=detailed_analysis
        )

    except Exception as e:
        logger.error(f"增強圖像分析錯誤: {str(e)}")
        return AIAnalysisResult(
            is_venue_detected=False,
            confidence=0.0,
            detected_objects=[],
            risk_level="low",
            detailed_analysis={"error": str(e)}
        )


@app.get("/", response_class=HTMLResponse)
async def home():
    """返回測試用的 HTML 界面"""
    html_content = '''
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <title>Olympic CamGuard - International Sports Surveillance System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --olympic-blue: #0081C8;
                --olympic-yellow: #FCB131;
                --olympic-black: #000000;
                --olympic-green: #00A651;
                --olympic-red: #EE334E;
                --primary-gradient: linear-gradient(135deg, #0081C8 0%, #00A651 100%);
                --secondary-gradient: linear-gradient(135deg, #FCB131 0%, #EE334E 100%);
                --bg-gradient: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                --card-shadow: 0 10px 40px rgba(0,0,0,0.1);
                --border-radius: 16px;
                --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg-gradient);
                min-height: 100vh;
                line-height: 1.6;
                color: #2d3748;
            }

            .main-container {
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            .header {
                background: var(--primary-gradient);
                color: white;
                padding: 2rem 0;
                text-align: center;
                position: relative;
                overflow: hidden;
            }

            .header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="3" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="20" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="40" cy="60" r="2.5" fill="rgba(255,255,255,0.1)"/><circle cx="90" cy="70" r="1.5" fill="rgba(255,255,255,0.1)"/><circle cx="10" cy="80" r="2" fill="rgba(255,255,255,0.1)"/></svg>');
                animation: float 20s linear infinite;
            }

            @keyframes float {
                0% { transform: translateX(-50px) translateY(-50px); }
                100% { transform: translateX(50px) translateY(50px); }
            }

            .header-content {
                position: relative;
                z-index: 1;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 2rem;
            }

            .logo {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 1rem;
                margin-bottom: 1rem;
            }

            .olympic-rings {
                display: flex;
                gap: 0.5rem;
                align-items: center;
            }

            .ring {
                width: 24px;
                height: 24px;
                border: 3px solid;
                border-radius: 50%;
                animation: pulse 2s ease-in-out infinite alternate;
            }

            .ring:nth-child(1) { border-color: var(--olympic-blue); animation-delay: 0s; }
            .ring:nth-child(2) { border-color: var(--olympic-yellow); animation-delay: 0.2s; }
            .ring:nth-child(3) { border-color: var(--olympic-black); animation-delay: 0.4s; }
            .ring:nth-child(4) { border-color: var(--olympic-green); animation-delay: 0.6s; }
            .ring:nth-child(5) { border-color: var(--olympic-red); animation-delay: 0.8s; }

            @keyframes pulse {
                0% { transform: scale(1); opacity: 0.8; }
                100% { transform: scale(1.1); opacity: 1; }
            }

            .main-title {
                font-family: 'Poppins', sans-serif;
                font-size: 3rem;
                font-weight: 700;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }

            .subtitle {
                font-size: 1.2rem;
                font-weight: 300;
                opacity: 0.95;
                margin-top: 0.5rem;
            }

            .container {
                flex: 1;
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 2rem;
            }

            .section {
                background: white;
                border-radius: var(--border-radius);
                padding: 2rem;
                box-shadow: var(--card-shadow);
                transition: var(--transition);
                border: 1px solid rgba(255,255,255,0.2);
                position: relative;
                overflow: hidden;
            }

            .section::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--primary-gradient);
            }

            .section:hover {
                transform: translateY(-8px);
                box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            }

            .section-header {
                display: flex;
                align-items: center;
                gap: 1rem;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 2px solid #f7fafc;
            }

            .section-icon {
                width: 48px;
                height: 48px;
                border-radius: 12px;
                background: var(--primary-gradient);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.5rem;
            }

            .section-title {
                font-family: 'Poppins', sans-serif;
                font-size: 1.4rem;
                font-weight: 600;
                color: #2d3748;
                margin: 0;
            }

            .button-group {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                margin-bottom: 1.5rem;
            }

            .btn {
                background: var(--primary-gradient);
                color: white;
                border: none;
                padding: 0.8rem 1.5rem;
                border-radius: 12px;
                font-weight: 500;
                cursor: pointer;
                transition: var(--transition);
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.9rem;
                min-height: 44px;
                box-shadow: 0 4px 12px rgba(0,129,200,0.3);
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,129,200,0.4);
            }

            .btn:active {
                transform: translateY(0);
            }

            .btn-secondary {
                background: var(--secondary-gradient);
                box-shadow: 0 4px 12px rgba(252,177,49,0.3);
            }

            .btn-secondary:hover {
                box-shadow: 0 8px 25px rgba(252,177,49,0.4);
            }

            .btn-outline {
                background: transparent;
                border: 2px solid var(--olympic-blue);
                color: var(--olympic-blue);
                box-shadow: none;
            }

            .btn-outline:hover {
                background: var(--olympic-blue);
                color: white;
            }

            .file-input-wrapper {
                position: relative;
                margin: 1rem 0;
            }

            .file-input {
                display: none;
            }

            .file-input-label {
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 1rem;
                border: 2px dashed #cbd5e0;
                border-radius: 12px;
                cursor: pointer;
                transition: var(--transition);
                background: #f7fafc;
            }

            .file-input-label:hover {
                border-color: var(--olympic-blue);
                background: rgba(0,129,200,0.05);
            }

            .result {
                margin-top: 1rem;
                padding: 1.5rem;
                border-radius: 12px;
                font-size: 0.9rem;
                line-height: 1.6;
                transition: var(--transition);
                min-height: 60px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }

            .status {
                padding: 1rem 1.5rem;
                border-radius: 12px;
                margin: 1rem 0;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 1rem;
                transition: var(--transition);
            }

            .safe {
                background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
                color: #22543d;
                border-left: 4px solid var(--olympic-green);
            }

            .warning {
                background: linear-gradient(135deg, #fffbf0 0%, #fed7aa 100%);
                color: #744210;
                border-left: 4px solid var(--olympic-yellow);
            }

            .danger {
                background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
                color: #742a2a;
                border-left: 4px solid var(--olympic-red);
            }

            .status-icon {
                font-size: 1.2rem;
            }

            .footer {
                background: #2d3748;
                color: white;
                text-align: center;
                padding: 2rem;
                margin-top: 3rem;
            }

            .footer-content {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 1rem;
            }

            .version-badge {
                background: var(--primary-gradient);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 500;
            }

            .loading {
                opacity: 0.6;
                pointer-events: none;
            }

            .loading::after {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 20px;
                height: 20px;
                margin: -10px 0 0 -10px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid var(--olympic-blue);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            @media (max-width: 768px) {
                .main-title {
                    font-size: 2rem;
                }
                
                .container {
                    grid-template-columns: 1fr;
                    padding: 1rem;
                }
                
                .section {
                    padding: 1.5rem;
                }
                
                .button-group {
                    flex-direction: column;
                }
                
                .footer-content {
                    flex-direction: column;
                    text-align: center;
                }
            }

            .pulse-animation {
                animation: statusPulse 2s ease-in-out infinite;
            }

            @keyframes statusPulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.02); }
            }

            /* Enhanced Notification System */
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                border-radius: 12px;
                padding: 1rem 1.5rem;
                box-shadow: var(--card-shadow);
                border-left: 4px solid var(--olympic-blue);
                z-index: 1000;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                min-width: 300px;
                animation: slideIn 0.3s ease-out;
                font-weight: 500;
            }

            .notification.success {
                border-left-color: var(--olympic-green);
                color: #22543d;
            }

            .notification.error {
                border-left-color: var(--olympic-red);
                color: #742a2a;
            }

            .notification.info {
                border-left-color: var(--olympic-blue);
                color: #2d3748;
            }

            @keyframes slideIn {
                0% {
                    transform: translateX(100%);
                    opacity: 0;
                }
                100% {
                    transform: translateX(0);
                    opacity: 1;
                }
            }

            /* Enhanced File Input */
            .file-input-label.active {
                border-color: var(--olympic-blue);
                background: rgba(0,129,200,0.1);
                transform: scale(1.02);
            }

            /* Enhanced Button States */
            .btn.processing {
                opacity: 0.7;
                pointer-events: none;
            }

            .btn.processing::after {
                content: '';
                width: 16px;
                height: 16px;
                border: 2px solid transparent;
                border-top: 2px solid white;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-left: 0.5rem;
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            <header class="header">
                <div class="header-content">
                    <div class="logo">
                        <div class="olympic-rings">
                            <div class="ring"></div>
                            <div class="ring"></div>
                            <div class="ring"></div>
                            <div class="ring"></div>
                            <div class="ring"></div>
                        </div>
                    </div>
                    <h1 class="main-title">Olympic CamGuard</h1>
                    <p class="subtitle">International Sports Venue AI Surveillance System</p>
                </div>
            </header>
            
            <div class="container">
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">
                            <i class="fas fa-map-marker-alt"></i>
                        </div>
                        <h3 class="section-title">GPS Location Monitoring</h3>
                    </div>
                    <div class="button-group">
                        <button class="btn" onclick="testLocation()">
                            <i class="fas fa-crosshairs"></i>
                            Get Current Location
                        </button>
                        <button class="btn btn-secondary" onclick="simulateVenueLocation()">
                            <i class="fas fa-building"></i>
                            Simulate Venue Area
                        </button>
                        <button class="btn btn-outline" onclick="simulateWarningLocation()">
                            <i class="fas fa-exclamation-triangle"></i>
                            Simulate Warning Zone
                        </button>
                    </div>
                    <div id="locationResult" class="result"></div>
                </div>
                
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">
                            <i class="fas fa-camera"></i>
                        </div>
                        <h3 class="section-title">Camera Control System</h3>
                    </div>
                    <div class="button-group">
                        <button class="btn" onclick="checkCameraStatus()">
                            <i class="fas fa-video"></i>
                            Check Camera Status
                        </button>
                        <button class="btn btn-secondary" onclick="simulatePhotoCapture()">
                            <i class="fas fa-camera-retro"></i>
                            Simulate Photo Capture
                        </button>
                    </div>
                    <div id="cameraResult" class="result"></div>
                </div>
                
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">
                            <i class="fas fa-brain"></i>
                        </div>
                        <h3 class="section-title">AI Image Analysis</h3>
                    </div>
                    <div class="file-input-wrapper">
                        <input type="file" id="imageInput" class="file-input" accept="image/*">
                        <label for="imageInput" class="file-input-label">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <span>Choose image file to analyze...</span>
                        </label>
                    </div>
                    <div class="button-group">
                        <button class="btn" onclick="analyzeImage()">
                            <i class="fas fa-search"></i>
                            Analyze Image
                        </button>
                    </div>
                    <div id="imageResult" class="result"></div>
                </div>
                
                <div class="section">
                    <div class="section-header">
                        <div class="section-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <h3 class="section-title">System Dashboard</h3>
                    </div>
                    <div class="button-group">
                        <button class="btn" onclick="getSystemStatus()">
                            <i class="fas fa-sync-alt"></i>
                            Refresh Status
                        </button>
                        <button class="btn btn-outline" onclick="getDetectionLogs()">
                            <i class="fas fa-history"></i>
                            View Logs
                        </button>
                    </div>
                    <div id="systemStatus" class="result"></div>
                </div>
            </div>
            
            <footer class="footer">
                <div class="footer-content">
                    <div>
                        <strong>Olympic CamGuard</strong> - Protecting International Sports Broadcasting Rights
                    </div>
                    <div class="version-badge">
                        Alpha v0.2.0
                    </div>
                </div>
            </footer>
        </div>

        <script>
            let currentDeviceId = 'olympic-device-' + Math.random().toString(36).substr(2, 9);
            
            // Enhanced loading states
            function setLoading(elementId, isLoading) {
                const element = document.getElementById(elementId);
                if (isLoading) {
                    element.classList.add('loading');
                    element.innerHTML = '<div style="text-align: center; padding: 2rem;"><i class="fas fa-spinner fa-spin"></i> Processing...</div>';
                } else {
                    element.classList.remove('loading');
                }
            }
            
            // Enhanced notifications
            function showNotification(message, type = 'info') {
                const notification = document.createElement('div');
                notification.className = `notification ${type}`;
                notification.innerHTML = `
                    <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
                    ${message}
                `;
                document.body.appendChild(notification);
                setTimeout(() => notification.remove(), 4000);
            }
            
            async function testLocation() {
                setLoading('locationResult', true);
                
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
                            setLoading('locationResult', false);
                            displayLocationResult(result);
                            showNotification('Location check completed', 'success');
                        } catch (error) {
                            setLoading('locationResult', false);
                            document.getElementById('locationResult').innerHTML = 
                                '<div class="status danger"><i class="fas fa-exclamation-triangle status-icon"></i>Location check failed: ' + error.message + '</div>';
                            showNotification('Location check failed', 'error');
                        }
                    }, (error) => {
                        setLoading('locationResult', false);
                        document.getElementById('locationResult').innerHTML = 
                            '<div class="status danger"><i class="fas fa-map-marker-alt status-icon"></i>Unable to get location: ' + error.message + '</div>';
                        showNotification('Geolocation access denied', 'error');
                    });
                } else {
                    setLoading('locationResult', false);
                    document.getElementById('locationResult').innerHTML = 
                        '<div class="status danger"><i class="fas fa-times-circle status-icon"></i>Browser does not support geolocation</div>';
                    showNotification('Geolocation not supported', 'error');
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
                let statusIcon = 'check-circle';
                if (result.risk_level === 'high') {
                    statusClass = 'danger';
                    statusIcon = 'exclamation-triangle';
                } else if (result.risk_level === 'medium') {
                    statusClass = 'warning';
                    statusIcon = 'exclamation-circle';
                }
                
                const resultElement = document.getElementById('locationResult');
                resultElement.innerHTML = 
                    '<div class="status ' + statusClass + ' pulse-animation">' +
                    '<i class="fas fa-' + statusIcon + ' status-icon"></i>' +
                    '<div>' +
                    '<strong>Location Status:</strong> ' + result.message + '<br>' +
                    '<strong>Distance to Nearest Venue:</strong> ' + Math.round(result.distance) + ' meters<br>' +
                    '<strong>Risk Level:</strong> ' + result.risk_level.toUpperCase() + '<br>' +
                    '<strong>Zone:</strong> ' + result.zone +
                    '</div>' +
                    '</div>';
                
                // Add pulse animation temporarily
                setTimeout(() => {
                    resultElement.querySelector('.status').classList.remove('pulse-animation');
                }, 2000);
            }
            
            async function checkCameraStatus() {
                setLoading('cameraResult', true);
                
                try {
                    const response = await fetch('/api/camera-status/' + currentDeviceId);
                    const result = await response.json();
                    
                    let statusClass = result.is_blocked ? 'danger' : 'safe';
                    let statusIcon = result.is_blocked ? 'ban' : 'video';
                    let statusText = result.is_blocked ? 'BLOCKED' : 'ACTIVE';
                    
                    setLoading('cameraResult', false);
                    document.getElementById('cameraResult').innerHTML = 
                        '<div class="status ' + statusClass + ' pulse-animation">' +
                        '<i class="fas fa-' + statusIcon + ' status-icon"></i>' +
                        '<div>' +
                        '<strong>Camera Status:</strong> ' + statusText + '<br>' +
                        '<strong>Reason:</strong> ' + result.reason + '<br>' +
                        '<strong>Device ID:</strong> ' + currentDeviceId +
                        '</div>' +
                        '</div>';
                    
                    showNotification('Camera status updated', 'success');
                    
                    setTimeout(() => {
                        document.getElementById('cameraResult').querySelector('.status').classList.remove('pulse-animation');
                    }, 2000);
                } catch (error) {
                    setLoading('cameraResult', false);
                    document.getElementById('cameraResult').innerHTML = 
                        '<div class="status danger"><i class="fas fa-exclamation-triangle status-icon"></i>Status check failed: ' + error.message + '</div>';
                    showNotification('Camera status check failed', 'error');
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
                    showNotification('Please select an image file first', 'error');
                    return;
                }
                
                setLoading('imageResult', true);
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    const response = await fetch('/api/analyze-image', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    
                    let statusClass = 'safe';
                    let statusIcon = 'check-circle';
                    if (result.risk_level === 'high') {
                        statusClass = 'danger';
                        statusIcon = 'exclamation-triangle';
                    } else if (result.risk_level === 'medium') {
                        statusClass = 'warning';
                        statusIcon = 'exclamation-circle';
                    }
                    
                    setLoading('imageResult', false);
                    document.getElementById('imageResult').innerHTML = 
                        '<div class="status ' + statusClass + ' pulse-animation">' +
                        '<i class="fas fa-' + statusIcon + ' status-icon"></i>' +
                        '<div>' +
                        '<strong>Venue Detection:</strong> ' + (result.is_venue_detected ? 'DETECTED' : 'NOT DETECTED') + '<br>' +
                        '<strong>Confidence:</strong> ' + Math.round(result.confidence * 100) + '%<br>' +
                        '<strong>Detected Objects:</strong> ' + (result.detected_objects.length > 0 ? result.detected_objects.join(', ') : 'None') + '<br>' +
                        '<strong>Risk Level:</strong> ' + result.risk_level.toUpperCase() + '<br>' +
                        '<strong>Features Found:</strong> ' + (result.detailed_analysis?.total_features || 0) +
                        '</div>' +
                        '</div>';
                    
                    showNotification('Image analysis completed', result.is_venue_detected ? 'error' : 'success');
                    
                    setTimeout(() => {
                        document.getElementById('imageResult').querySelector('.status').classList.remove('pulse-animation');
                    }, 2000);
                } catch (error) {
                    setLoading('imageResult', false);
                    document.getElementById('imageResult').innerHTML = 
                        '<div class="status danger"><i class="fas fa-exclamation-triangle status-icon"></i>Image analysis failed: ' + error.message + '</div>';
                    showNotification('Image analysis failed', 'error');
                }
            }
            
            async function getSystemStatus() {
                setLoading('systemStatus', true);
                
                try {
                    const response = await fetch('/api/system-status');
                    const result = await response.json();
                    
                    setLoading('systemStatus', false);
                    document.getElementById('systemStatus').innerHTML = 
                        '<div class="status safe pulse-animation">' +
                        '<i class="fas fa-server status-icon"></i>' +
                        '<div>' +
                        '<strong>System Status:</strong> ' + result.uptime + '<br>' +
                        '<strong>Blocked Devices:</strong> ' + result.blocked_devices_count + '<br>' +
                        '<strong>Detection Logs:</strong> ' + result.detection_logs_count + '<br>' +
                        '<strong>Active Venues:</strong> ' + result.active_venues.join(', ') + '<br>' +
                        '<strong>Geofence Radius:</strong> ' + result.geofence_radius + 'm' +
                        '</div>' +
                        '</div>';
                    
                    showNotification('System status refreshed', 'success');
                    
                    setTimeout(() => {
                        document.getElementById('systemStatus').querySelector('.status').classList.remove('pulse-animation');
                    }, 2000);
                } catch (error) {
                    setLoading('systemStatus', false);
                    document.getElementById('systemStatus').innerHTML = 
                        '<div class="status danger"><i class="fas fa-exclamation-triangle status-icon"></i>Status retrieval failed: ' + error.message + '</div>';
                    showNotification('System status check failed', 'error');
                }
            }
            
            async function getDetectionLogs() {
                setLoading('systemStatus', true);
                
                try {
                    const response = await fetch('/api/detection-logs?limit=10');
                    const result = await response.json();
                    
                    setLoading('systemStatus', false);
                    
                    let logsHtml = '<div class="status safe"><i class="fas fa-history status-icon"></i><div>';
                    logsHtml += '<strong>Recent Detection Logs (' + result.total + ' total):</strong><br><br>';
                    
                    if (result.logs.length > 0) {
                        result.logs.reverse().forEach(log => {
                            const timestamp = new Date(log.timestamp).toLocaleString();
                            logsHtml += '<div style="margin-bottom: 8px; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 6px;">';
                            logsHtml += '<strong>' + timestamp + '</strong><br>';
                            if (log.type === 'image_analysis') {
                                logsHtml += 'Image Analysis - ' + log.filename + '<br>';
                                logsHtml += 'Venue Detected: ' + (log.result.is_venue_detected ? 'Yes' : 'No');
                            } else {
                                logsHtml += 'Location Check - ' + log.device_id + '<br>';
                                logsHtml += 'Zone: ' + log.zone + ', Distance: ' + Math.round(log.distance) + 'm';
                            }
                            logsHtml += '</div>';
                        });
                    } else {
                        logsHtml += '<em>No logs available</em>';
                    }
                    
                    logsHtml += '</div></div>';
                    document.getElementById('systemStatus').innerHTML = logsHtml;
                    
                    showNotification('Detection logs loaded', 'info');
                } catch (error) {
                    setLoading('systemStatus', false);
                    document.getElementById('systemStatus').innerHTML = 
                        '<div class="status danger"><i class="fas fa-exclamation-triangle status-icon"></i>Logs retrieval failed: ' + error.message + '</div>';
                    showNotification('Failed to load logs', 'error');
                }
            }
            
            // Enhanced file input interaction
            document.addEventListener('DOMContentLoaded', function() {
                const fileInput = document.getElementById('imageInput');
                const fileLabel = document.querySelector('.file-input-label');
                
                fileInput.addEventListener('change', function() {
                    if (this.files[0]) {
                        fileLabel.querySelector('span').textContent = 'Selected: ' + this.files[0].name;
                        fileLabel.classList.add('active');
                        showNotification('Image selected: ' + this.files[0].name, 'info');
                    }
                });
                
                // Initialize system status
                getSystemStatus();
                
                // Auto-refresh system status every 30 seconds
                setInterval(getSystemStatus, 30000);
            });
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)


@app.post("/api/check-location")
async def check_location(location: LocationData):
    """檢查GPS位置並判斷是否在限制區域"""
    try:
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

        # 確保距離是有限數值
        safe_distance = min(distance, 999999) if distance != float(
            'inf') else 999999

        # 記錄檢測
        detection_logs.append({
            "timestamp": location.timestamp,
            "device_id": location.device_id,
            "location": (location.latitude, location.longitude),
            "zone": zone_name,
            "distance": safe_distance,
            "risk_level": risk_level
        })

        return {
            "message": message,
            "risk_level": risk_level,
            "zone": zone_name,
            "distance": safe_distance,
            "camera_blocked": is_restricted
        }

    except Exception as e:
        logger.error(f"位置檢查錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"位置檢查失敗: {str(e)}")


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
