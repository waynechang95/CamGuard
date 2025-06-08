#!/usr/bin/env python3
"""
Olympic CamGuard - 改進的圖像分析模組
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    feature_type: str
    confidence: float
    description: str = ""


class ImprovedVenueAnalyzer:
    """改進的場館分析器"""

    def __init__(self):
        # 奧運相關顏色定義 (HSV)
        self.olympic_colors = {
            "blue": [(100, 50, 50), (130, 255, 255)],
            "yellow": [(20, 100, 100), (30, 255, 255)],
            "green": [(50, 100, 50), (70, 255, 255)],
            "red": [(0, 120, 70), (10, 255, 255)],
            "black": [(0, 0, 0), (180, 255, 30)]
        }

        # 場館關鍵詞 (可擴展為 OCR)
        self.venue_keywords = [
            "olympic", "stadium", "arena", "sports", "games",
            "奧運", "體育場", "競技場", "運動", "比賽"
        ]

    def analyze_image_improved(self, image_data: bytes) -> Dict:
        """改進的圖像分析主函數"""
        try:
            # 解碼圖像
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("無法解析圖像")

            # 多層檢測
            results = []

            # 1. 建築結構檢測 (改進版)
            results.append(self._detect_architecture_advanced(img))

            # 2. 奧運色彩檢測 (改進版)
            results.append(self._detect_olympic_colors_advanced(img))

            # 3. 體育場特徵檢測
            results.append(self._detect_stadium_features(img))

            # 4. 人群和活動檢測
            results.append(self._detect_crowd_activity(img))

            # 5. 標誌和文字檢測
            results.append(self._detect_signage_advanced(img))

            # 6. 幾何形狀檢測
            results.append(self._detect_geometric_patterns(img))

            # 編譯最終結果
            return self._compile_final_result(results, img.shape)

        except Exception as e:
            logger.error(f"改進圖像分析錯誤: {str(e)}")
            return self._get_error_result()

    def _detect_architecture_advanced(self, img: np.ndarray) -> DetectionResult:
        """進階建築檢測"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 多尺度邊緣檢測
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 100, 200)
        edges_combined = cv2.bitwise_or(edges1, edges2)

        # 檢測直線
        lines = cv2.HoughLinesP(
            edges_combined, 1, np.pi/180, threshold=50,
            minLineLength=50, maxLineGap=10
        )

        confidence = 0.0
        features = []

        if lines is not None:
            # 分析線條特徵
            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi

                if length > 100:  # 只考慮長線條
                    if abs(angle) < 15 or abs(angle) > 165:
                        horizontal_lines.append(line)
                    elif 75 < abs(angle) < 105:
                        vertical_lines.append(line)

            # 建築物特徵評分
            if len(horizontal_lines) > 3:
                features.append("水平結構線")
                confidence += 0.2

            if len(vertical_lines) > 3:
                features.append("垂直結構線")
                confidence += 0.2

            # 檢查線條規律性
            if len(horizontal_lines) > 5 and len(vertical_lines) > 5:
                features.append("規律建築結構")
                confidence += 0.3

        # 檢測大型輪廓
        contours, _ = cv2.findContours(
            edges_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > 5000]

        if large_contours:
            features.append(f"{len(large_contours)}個大型結構")
            confidence += min(len(large_contours) * 0.1, 0.3)

        return DetectionResult(
            feature_type="advanced_architecture",
            confidence=min(confidence, 0.7),
            description=f"建築特徵: {', '.join(features) if features else '無'}"
        )

    def _detect_olympic_colors_advanced(self, img: np.ndarray) -> DetectionResult:
        """進階奧運色彩檢測"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        total_pixels = img.shape[0] * img.shape[1]

        color_scores = {}
        detected_colors = []

        for color_name, (lower, upper) in self.olympic_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_pixels = cv2.countNonZero(mask)
            color_ratio = color_pixels / total_pixels

            if color_ratio > 0.05:  # 至少佔5%
                color_scores[color_name] = color_ratio
                detected_colors.append(color_name)

        # 計算色彩多樣性分數
        confidence = 0.0
        if len(detected_colors) >= 2:
            confidence = min(len(detected_colors) * 0.15, 0.6)

            # 如果檢測到奧運五環的顏色組合
            olympic_ring_colors = {"blue", "yellow", "green", "red", "black"}
            detected_set = set(detected_colors)
            overlap = len(detected_set.intersection(olympic_ring_colors))

            if overlap >= 3:
                confidence += 0.2

        return DetectionResult(
            feature_type="olympic_colors",
            confidence=confidence,
            description=f"檢測到顏色: {', '.join(detected_colors) if detected_colors else '無'}"
        )

    def _detect_stadium_features(self, img: np.ndarray) -> DetectionResult:
        """體育場特徵檢測"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        confidence = 0.0
        features = []

        # 1. 圓形/橢圓檢測 (跑道、體育場)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30, minRadius=20, maxRadius=200
        )

        if circles is not None:
            circle_count = len(circles[0])
            if circle_count >= 1:
                features.append(f"{circle_count}個圓形結構")
                confidence += min(circle_count * 0.2, 0.4)

        # 2. 檢測運動場標線 (白線)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))

        # 形態學操作增強線條
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        white_lines = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        line_contours, _ = cv2.findContours(
            white_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        long_lines = [c for c in line_contours if cv2.contourArea(c) > 200]

        if long_lines:
            features.append(f"{len(long_lines)}條標線")
            confidence += min(len(long_lines) * 0.1, 0.3)

        # 3. 檢測看台結構 (階梯狀)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
                                minLineLength=30, maxLineGap=5)

        if lines is not None:
            # 檢測平行線 (看台特徵)
            parallel_groups = self._find_parallel_lines(lines)
            if len(parallel_groups) > 2:
                features.append("看台階梯結構")
                confidence += 0.25

        return DetectionResult(
            feature_type="stadium_features",
            confidence=min(confidence, 0.8),
            description=f"體育場特徵: {', '.join(features) if features else '無'}"
        )

    def _detect_crowd_activity(self, img: np.ndarray) -> DetectionResult:
        """人群和活動檢測"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        confidence = 0.0
        indicators = []

        # 1. 膚色檢測
        skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
        skin_ratio = cv2.countNonZero(
            skin_mask) / (img.shape[0] * img.shape[1])

        if 0.01 < skin_ratio < 0.15:  # 適量膚色表示人群
            indicators.append("人群膚色")
            confidence += 0.2

        # 2. 色彩多樣性 (人群服裝)
        color_diversity = self._calculate_color_diversity(hsv)
        if color_diversity > 0.4:
            indicators.append("服裝色彩多樣")
            confidence += 0.15

        # 3. 運動模糊檢測 (活動指標)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < 100:  # 低方差可能表示運動模糊
            indicators.append("運動模糊")
            confidence += 0.1

        return DetectionResult(
            feature_type="crowd_activity",
            confidence=min(confidence, 0.45),
            description=f"活動指標: {', '.join(indicators) if indicators else '無'}"
        )

    def _detect_signage_advanced(self, img: np.ndarray) -> DetectionResult:
        """進階標誌檢測"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        confidence = 0.0
        features = []

        # 1. 文字區域檢測
        # 使用 MSER (Maximally Stable Extremal Regions) 檢測文字
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        text_regions = 0
        for region in regions:
            if 50 < len(region) < 2000:  # 文字區域大小範圍
                text_regions += 1

        if text_regions > 5:
            features.append(f"{text_regions}個文字區域")
            confidence += min(text_regions * 0.02, 0.3)

        # 2. 矩形標誌檢測
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangular_signs = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:
                # 檢查是否接近矩形
                approx = cv2.approxPolyDP(
                    contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    rectangular_signs += 1

        if rectangular_signs > 0:
            features.append(f"{rectangular_signs}個矩形標誌")
            confidence += min(rectangular_signs * 0.1, 0.2)

        return DetectionResult(
            feature_type="signage",
            confidence=min(confidence, 0.5),
            description=f"標誌特徵: {', '.join(features) if features else '無'}"
        )

    def _detect_geometric_patterns(self, img: np.ndarray) -> DetectionResult:
        """幾何圖案檢測"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        confidence = 0.0
        patterns = []

        # 1. 對稱性檢測
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)

        # 調整大小以匹配
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]

        # 計算相似度
        similarity = cv2.matchTemplate(
            left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]

        if similarity > 0.7:
            patterns.append("水平對稱")
            confidence += 0.2

        # 2. 重複圖案檢測
        # 使用模板匹配檢測重複結構
        template_size = min(50, width//4, height//4)
        if template_size > 20:
            template = gray[:template_size, :template_size]
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.8)

            if len(locations[0]) > 3:  # 找到多個匹配
                patterns.append("重複圖案")
                confidence += 0.15

        return DetectionResult(
            feature_type="geometric_patterns",
            confidence=min(confidence, 0.35),
            description=f"幾何圖案: {', '.join(patterns) if patterns else '無'}"
        )

    def _find_parallel_lines(self, lines: np.ndarray) -> List[List]:
        """尋找平行線組"""
        if lines is None:
            return []

        parallel_groups = []
        angle_threshold = 10  # 角度容差

        for i, line1 in enumerate(lines):
            x1, y1, x2, y2 = line1[0]
            angle1 = np.arctan2(y2-y1, x2-x1) * 180 / np.pi

            group = [line1]
            for j, line2 in enumerate(lines[i+1:], i+1):
                x3, y3, x4, y4 = line2[0]
                angle2 = np.arctan2(y4-y3, x4-x3) * 180 / np.pi

                if abs(angle1 - angle2) < angle_threshold:
                    group.append(line2)

            if len(group) > 1:
                parallel_groups.append(group)

        return parallel_groups

    def _calculate_color_diversity(self, hsv_img: np.ndarray) -> float:
        """計算色彩多樣性"""
        hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
        non_zero_bins = np.count_nonzero(hist)
        return non_zero_bins / 180.0

    def _compile_final_result(self, results: List[DetectionResult], img_shape: Tuple) -> Dict:
        """編譯最終結果"""
        total_confidence = 0.0
        detected_objects = []
        detailed_analysis = {}

        # 加權計算總信心度
        weights = {
            "advanced_architecture": 1.2,
            "stadium_features": 1.5,
            "olympic_colors": 1.0,
            "crowd_activity": 0.8,
            "signage": 1.0,
            "geometric_patterns": 0.7
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for result in results:
            if result.confidence > 0.05:
                weight = weights.get(result.feature_type, 1.0)
                weighted_sum += result.confidence * weight
                total_weight += weight

                detected_objects.append(result.feature_type)
                detailed_analysis[result.feature_type] = {
                    'confidence': result.confidence,
                    'description': result.description,
                    'weight': weight
                }

        # 計算加權平均信心度
        final_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        final_confidence = min(final_confidence, 1.0)

        # 判斷檢測結果
        is_venue_detected = final_confidence > 0.5

        # 風險等級
        if final_confidence > 0.8:
            risk_level = "high"
        elif final_confidence > 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            'is_venue_detected': is_venue_detected,
            'confidence': round(final_confidence, 3),
            'detected_objects': detected_objects,
            'risk_level': risk_level,
            'detailed_analysis': detailed_analysis,
            'image_size': f"{img_shape[1]}x{img_shape[0]}",
            'analysis_method': 'improved_multi_layer'
        }

    def _get_error_result(self) -> Dict:
        """錯誤回退結果"""
        return {
            'is_venue_detected': False,
            'confidence': 0.0,
            'detected_objects': [],
            'risk_level': 'low',
            'detailed_analysis': {},
            'error': 'Analysis failed'
        }

# 使用函數


def analyze_image_improved(image_data: bytes) -> Dict:
    """改進的圖像分析函數"""
    analyzer = ImprovedVenueAnalyzer()
    return analyzer.analyze_image_improved(image_data)
